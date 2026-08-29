# OCT-Based Retinal Disease Classification and AI Clinical Decision Support

8-class retinal OCT B-scan classifier (**Retinal OCT-C8**) with a deterministic
clinical-decision-support layer that turns calibrated model probabilities into a
triage recommendation and a human-readable report.

> Research / decision-support tooling. **Not a diagnostic device.** Every
> recommendation is for review by a qualified clinician.

## Classes

`AMD` · `CNV` · `CSR` · `DME` · `DR` · `Drusen` · `Macular Hole` · `Normal`
(frozen ids in [`data/metadata/label_map.json`](data/metadata/label_map.json))

## Data

| Dataset | Role | Notes |
|---|---|---|
| Retinal OCT-C8 (Kaggle `obulisainaren/retinal-oct-c8`) | train / val / test | Authors' pre-split, 3,000/class → 18,400 / 2,800 / 2,800. **Not re-split.** |
| OPTOPOL REVO clinic B-scans (`data/external/clinic_optopol/`, 37 images) | external validation only | Different vendor. Never trained on, never used for model selection. |

`data/` is git-ignored — patient scans never leave this machine.

## Layout

```
configs/            Hydra configs (data/ model/ training/ preprocess/ cds/)
data/metadata/      label_map.json + data dictionary (only tracked data files)
src/oct_cds/
  data/             manifests, OPTOPOL filename parser, Dataset + DataModule, QC
  preprocessing/    deterministic transforms + train-only augmentation
  models/           timm backbones, LightningModule, losses, temperature scaling
  evaluation/       metrics (sens/spec, AUROC/AUPRC, QWK, ECE), bootstrap CIs
  explainability/   Grad-CAM overlays for the report
  cds/              schema, rule engine, OOD gate, guideline refs, report, audit
train.py            Hydra training entrypoint
tests/
```

## Quickstart

```bash
pip install -e ".[dev,explain]"
```

```bash
# 1. put OCT-C8 under data/raw/oct_c8/{train,val,test}/<CLASS>/  and the
#    clinic scans under data/external/clinic_optopol/
# 2. build manifests (writes data/processed/*.csv)
python -m oct_cds.cli data build
```

```bash
# 3. train (DenseNet-121 primary; swap backbone from the CLI)
python train.py model=densenet121 data=oct_c8
python train.py model=resnet50
python train.py model=efficientnet_b3 training.max_epochs=40
```

```bash
# 4. see the CDS rule engine on a synthetic case
python -m oct_cds.cli cds demo
```

```bash
pytest
```

## Pipeline stages

1. **Ingest & QC** — checksum, de-identification review, quality flags (`src/oct_cds/data/`)
2. **Manifests** — CSV per split; patient-level leakage checks; OPTOPOL locked to `external_test`
3. **Preprocess** — ROI crop → resize → grayscale→RGB → ImageNet normalize (identical for every split)
4. **Train** — timm backbone, head-warmup then unfreeze, weighted/focal loss option, cosine LR
5. **Calibrate** — temperature scaling on val; CDS consumes calibrated probs only
6. **Evaluate** — internal test + OPTOPOL external set; per-class sens/spec, AUROC/AUPRC, QWK, ECE, bootstrap CIs
7. **Explain** — Grad-CAM overlays
8. **CDS** — OOD gate → confidence/margin abstention → rule-based urgency → report + audit log
