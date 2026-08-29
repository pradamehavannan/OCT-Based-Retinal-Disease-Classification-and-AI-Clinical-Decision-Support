# Figures used in the README

These are copied out of the (git-ignored) run outputs so the README renders on
GitHub. Regenerate the source overlays with:

```bash
python explain.py paths=<env> explain.split=external_test \
    'explain.classes=[Drusen]' explain.only_errors=true explain.target=both
```

That writes overlays to
`<output_dir>/explain/external_test/wrong/Drusen/`, each a
`[ raw B-scan | Grad-CAM predicted class | Grad-CAM true class ]` strip, plus an
`index.csv`.

## What the overlays show

Reviewing all 4 Drusen misclassification pairs (DRUSEN_3, DRUSEN_4, DRUSEN_7,
DRUSEN_9): in **3 of 4** the predicted-class heatmap and the true-class heatmap
land in **nearly the same location** — the model localises the pathological
region correctly but misreads *what it is*. Only DRUSEN_9__L (predicted DME)
showed genuinely divergent attention. So the dominant pattern is
**"correct localisation, wrong classification,"** not "looking in the wrong
place." This matters for the CDS discussion: the failure is semantic, not
spatial, so a saliency check would not catch it.

Context: **8 of 14** Drusen misclassifications went to `Normal` — the model
often finds the drusen region but reads it as unremarkable.

## Selecting the two figures

Pick two strips that are **representative of the dominant pattern** — predicted-
and true-class heatmaps in nearly the same spot, class flipped. Prefer the
"→ Normal" cases since that is the largest error bucket. The two used:

```bash
cp "<output_dir>/explain/external_test/wrong/Drusen/<file-1>.png" docs/figures/gradcam_drusen_wrong_1.png
cp "<output_dir>/explain/external_test/wrong/Drusen/<file-2>.png" docs/figures/gradcam_drusen_wrong_2.png
git add docs/figures/gradcam_drusen_wrong_1.png docs/figures/gradcam_drusen_wrong_2.png
```

| README figure | scan | true → pred | pred conf | heatmap pattern |
|---|---|---|---|---|
| `gradcam_drusen_wrong_1.png` | `DRUSEN_3__L` | Drusen → Normal | 0.88 | pred & true heatmaps ~co-located |
| `gradcam_drusen_wrong_2.png` | `DRUSEN_4__L` | Drusen → CSR | 0.54 | pred & true heatmaps ~co-located |

Counter-example (not used as a figure, but worth a sentence in the write-up):
`DRUSEN_9__L`, Drusen → DME — the one case with genuinely divergent attention.

> Fill in the exact source overlay filenames (they include the confidence in the
> name) once copied, e.g. `DRUSEN_3__L__pred-Normal_p0.88__cam-pred-Normal.png`.
