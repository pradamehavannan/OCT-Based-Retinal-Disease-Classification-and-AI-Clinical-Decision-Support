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

Pick an environment via the `paths` config group — `configs/paths/<env>.yaml`
holds the dataset + output locations for that machine:

| env | file | OCT-C8 / clinic roots | outputs |
|---|---|---|---|
| `default` | `paths/default.yaml` | Colab + Google Drive | `/content/drive/MyDrive/oct_cds_outputs` |
| `kaggle` | `paths/kaggle.yaml` | `/kaggle/input/...` (read-only) | `/kaggle/working/oct_cds_outputs` |

```bash
# 1. build manifests (writes data/processed/*.csv). --paths selects the env;
#    --set overrides any single path.
python -m oct_cds.cli data build --paths kaggle
#    or:  python -m oct_cds.cli data build --set paths.oct_c8_raw_root=/some/path
```

```bash
# 2. train (DenseNet-121 primary; swap backbone / env from the CLI)
python train.py paths=kaggle
python train.py paths=kaggle model=resnet50
python train.py model=efficientnet_b3 training.max_epochs=40   # default env
```

> **Resume:** `checkpoints/last.ckpt` is written every epoch under
> `paths.outputs_root`. Re-running the **same command** after a crash/disconnect
> **auto-resumes** from it (`training.auto_resume=false` to disable,
> `training.resume_from=<path>` to pick one). On Kaggle, `/kaggle/working` persists
> with the notebook version; on Colab, outputs default to Google Drive.
>
> **Colab:** `training.num_workers` defaults to 2 (~2 vCPUs). For speed, copy
> OCT-C8 off the Drive FUSE mount to `/content/oct_c8` first and pass
> `paths.oct_c8_raw_root=/content/oct_c8`. `train.py` hard-exits when run as
> `!python train.py` so the cell doesn't hang on lingering DataLoader workers.

```bash
# 3. evaluate on the held-out test set (auto-picks the best checkpoint under
#    <output_dir>/checkpoints, refits temperature scaling on val)
python eval.py paths=kaggle
python eval.py paths=kaggle eval.split=external_test          # the 37 OPTOPOL scans
python eval.py paths=kaggle eval.ckpt_path=/path/to/some.ckpt eval.calibration=load
```
> Writes `<output_dir>/eval/metrics_<split>.json` + `confusion_<split>.csv` and
> prints accuracy, macro/weighted/balanced F1, per-class sensitivity / specificity
> / precision / F1 / AUROC / AUPRC, macro AUROC/AUPRC, quadratic-weighted kappa,
> the confusion matrix, and ECE **before and after** temperature scaling.

```bash
# 4. Grad-CAM overlays  (needs the explain extra: pip install -e '.[explain]')
python explain.py paths=kaggle                                   # test set, <=200 overlays
python explain.py paths=kaggle explain.split=external_test       # the 37 clinic scans
python explain.py paths=kaggle explain.split=external_test \
    'explain.classes=[Drusen]' explain.only_errors=true explain.target=both
```
> Overlays: `<output_dir>/explain/<split>/<correct|wrong>/<true_class>/<stem>__pred-<x>_p<conf>__cam-<pred|true>-<class>.png`
> (each is `[ raw scan | heatmap overlay ]` on the model's 224px view). Plus
> `index.csv` mapping every overlay to true / predicted / probability. So the
> misclassified Drusen scans land in `.../explain/external_test/wrong/Drusen/`.
> `explain.target=both` saves a heatmap for the predicted class *and* the true
> class side by side — useful for seeing what the model latched onto instead of
> the drusen.

```bash
# 5. CDS layer — run calibrated predictions through the rules/urgency/abstention
python cds.py paths=kaggle                          # OCT-C8 test set
python cds.py paths=kaggle cds_run.split=external_test   # the 37 clinic scans
python cds.py paths=kaggle cds_run.split=external_test cds_run.write_reports_for=all
python -m oct_cds.cli cds demo                      # single synthetic case
```
> Writes to `<output_dir>/cds/`: `recommendations_<split>.csv` (per-image: probs,
> ood score, abstained / ood_rejected, urgency, recommendation text),
> `summary_<split>.json`, `reports/<split>/<stem>.{txt,json}` (full narratives for
> `write_reports_for`), and `audit_<split>.jsonl`. The summary's headline number:
> **on the images the model got wrong, how many did CDS defer to a specialist
> (good) vs assert a confident wrong urgency (bad)** — broken down by true class,
> so you can read off exactly how the misclassified Drusen cases were triaged.
> Tune thresholds on the CLI: `cds.min_confidence=0.75 cds.min_margin=0.20`.

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
