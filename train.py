"""Hydra entrypoint for training.

    python train.py                              # densenet121 on oct_c8 (defaults)
    python train.py model=resnet50
    python train.py model=efficientnet_b3 data=oct_c8 training.max_epochs=40
    python train.py -m model=densenet121,resnet50,efficientnet_b3   # sweep

Requires manifests to exist:  python -m oct_cds.cli data build
"""

from __future__ import annotations

import sys
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, str(Path(__file__).parent / "src"))

from oct_cds.common.logging import get_logger  # noqa: E402
from oct_cds.common.seed import seed_everything  # noqa: E402

log = get_logger("train")


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float:
    seed_everything(cfg.seed, deterministic=cfg.training.get("deterministic", True))
    log.info("\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    from oct_cds.data.dataset import make_datamodule
    from oct_cds.models.classifier import OCTClassifier

    data_cfg = OmegaConf.to_container(cfg.data, resolve=True)
    pre_cfg = OmegaConf.to_container(cfg.preprocess, resolve=True)
    train_cfg = OmegaConf.to_container(cfg.training, resolve=True)
    model_cfg = OmegaConf.to_container(cfg.model, resolve=True)

    manifest = Path(data_cfg["manifest"]["train"])
    if not manifest.exists():
        raise SystemExit(
            f"missing manifest {manifest}. Run: python -m oct_cds.cli data build"
        )

    # optional class weights
    class_weights = None
    if train_cfg.get("loss", {}).get("class_weighted"):
        from oct_cds.models.losses import class_weights_from_manifest

        class_weights = class_weights_from_manifest(str(manifest), model_cfg["num_classes"])

    dm = make_datamodule(data_cfg, pre_cfg, train_cfg)
    dm.setup("fit")
    model = OCTClassifier(model_cfg, train_cfg, class_weights=class_weights)

    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import (
        EarlyStopping,
        LearningRateMonitor,
        ModelCheckpoint,
    )

    ckpt_cfg = train_cfg.get("checkpoint", {})
    es_cfg = train_cfg.get("early_stopping", {})
    ckpt = ModelCheckpoint(
        dirpath=Path(cfg.output_dir) / "checkpoints",
        filename="{epoch}-{val/macro_f1:.4f}",
        monitor=ckpt_cfg.get("monitor", "val/macro_f1"),
        mode=ckpt_cfg.get("mode", "max"),
        save_top_k=int(ckpt_cfg.get("save_top_k", 2)),
    )
    callbacks = [
        ckpt,
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(
            monitor=es_cfg.get("monitor", "val/macro_f1"),
            mode=es_cfg.get("mode", "max"),
            patience=int(es_cfg.get("patience", 7)),
        ),
    ]

    trainer = pl.Trainer(
        max_epochs=int(train_cfg["max_epochs"]),
        precision=train_cfg.get("precision", "32-true"),
        accelerator=train_cfg.get("accelerator", "auto"),
        devices=train_cfg.get("devices", "auto"),
        deterministic=bool(train_cfg.get("deterministic", True)),
        default_root_dir=cfg.output_dir,
        callbacks=callbacks,
    )
    trainer.fit(model, dm.train_dataloader(), dm.val_dataloader())

    best_f1 = float(ckpt.best_model_score) if ckpt.best_model_score is not None else 0.0
    log.info("best val/macro_f1 = %.4f (%s)", best_f1, ckpt.best_model_path)

    # post-fit calibration on val
    if train_cfg.get("calibrate_after_fit"):
        _calibrate(model, dm, cfg)

    return best_f1


def _calibrate(model, dm, cfg) -> None:
    import torch

    from oct_cds.models.calibration import TemperatureScaler

    model.eval()
    logits, labels = [], []
    with torch.no_grad():
        for batch in dm.val_dataloader():
            logits.append(model(batch["image"]))
            labels.append(batch["label"])
    scaler = TemperatureScaler().fit(torch.cat(logits), torch.cat(labels))
    dest = Path(cfg.output_dir) / "calibrators" / "temperature.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    scaler.save(dest)
    log.info("fitted temperature = %.3f -> %s", scaler.temperature, dest)


if __name__ == "__main__":
    main()
