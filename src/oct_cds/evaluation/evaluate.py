"""Run a trained checkpoint over a split (internal ``test`` or the OPTOPOL
``external_test``) and write a metrics JSON to reports/metrics/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from oct_cds.common.logging import get_logger
from oct_cds.data.label_map import load_label_map
from oct_cds.evaluation.metrics import bootstrap_ci, classification_report_dict

log = get_logger(__name__)


def collect_predictions(model, dataloader, calibrator=None) -> dict[str, np.ndarray]:
    import torch

    model.eval()
    logits_all, y_all, paths = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch["image"].to(next(model.parameters()).device))
            logits_all.append(logits.cpu())
            y_all.append(batch["label"])
            paths.extend(batch["image_path"])
    logits = torch.cat(logits_all)
    y = torch.cat(y_all).numpy()
    probs = (
        calibrator.transform(logits).numpy()
        if calibrator is not None
        else torch.softmax(logits, dim=1).numpy()
    )
    return {"probs": probs, "y_true": y, "y_pred": probs.argmax(1), "paths": np.array(paths)}


def evaluate_split(
    model,
    dataloader,
    split_name: str,
    out_dir: str | Path,
    calibrator=None,
    bootstrap: bool = True,
) -> dict[str, Any]:
    lm = load_label_map()
    preds = collect_predictions(model, dataloader, calibrator)
    report = classification_report_dict(
        preds["y_true"], preds["y_pred"], preds["probs"], class_names=lm.keys
    )
    if bootstrap and len(preds["y_true"]):
        report["accuracy_ci"] = bootstrap_ci(
            lambda a, b: float((a == b).mean()), preds["y_true"], preds["y_pred"]
        )
    report["split"] = split_name

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"metrics_{split_name}.json"
    dest.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("wrote %s (macro_f1=%.4f)", dest, report["macro_f1"])
    return report
