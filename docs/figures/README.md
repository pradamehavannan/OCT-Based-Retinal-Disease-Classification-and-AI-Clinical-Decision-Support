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

Then copy two representative misclassified-Drusen strips here:

```bash
cp "<output_dir>/explain/external_test/wrong/Drusen/<file-A>.png" docs/figures/gradcam_drusen_wrong_1.png
cp "<output_dir>/explain/external_test/wrong/Drusen/<file-B>.png" docs/figures/gradcam_drusen_wrong_2.png
git add docs/figures/gradcam_drusen_wrong_1.png docs/figures/gradcam_drusen_wrong_2.png
```

Selection: pick two where the predicted-class heatmap and the true-class heatmap
attend to visibly different regions (that contrast is the point). Note the source
filenames here for traceability:

| README figure | source overlay | true → pred | conf |
|---|---|---|---|
| `gradcam_drusen_wrong_1.png` | _fill in_ | Drusen → _?_ | _?_ |
| `gradcam_drusen_wrong_2.png` | _fill in_ | Drusen → _?_ | _?_ |
