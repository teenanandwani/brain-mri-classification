# Run and understand the classification experiments — beginner's guide

This repo is a trimmed version of the original package. Keep your original Kaggle notebooks and your original project directory as backups. The original notebook and the segmentation project are intentionally **not** included in this classification repository.

## Before anything else — keep the baseline

Your original local folder was `~/Projects/brain-mri-portfolio/brain-mri-classification/` and your real run produced `outputs/`. We will **copy**, not move or delete it.

1. Unzip `brain-mri-classification-clean.zip` so that the new folder is `~/Projects/brain-mri-classification-clean`.
2. Open Terminal and enter:

```bash
cd ~/Projects/brain-mri-classification-clean
mkdir -p experiments
cp -R ~/Projects/brain-mri-portfolio/brain-mri-classification/outputs experiments/baseline
ls experiments/baseline
```

Expected key files: `best_model.keras`, `train_manifest.csv`, `val_manifest.csv`, `test_manifest.csv`, `metrics.json`, `test_predictions.csv`. This `experiments/` folder is ignored by Git and stays on your Mac. Do not copy or post CSVs containing `/Users/...` paths to a public repository.

## 1 — Inspect the six errors

Activate the environment you already used:

```bash
source ~/Projects/brain-mri-portfolio/brain-mri-classification/.venv/bin/activate
python scripts/review_errors.py --csv experiments/baseline/test_predictions.csv
open experiments/baseline/error_review.png
```

Look at the six errors and write short observations: visible text? unusual acquisition or contrast? strong borders? very small or cropped field of view? If a label looks questionable, **record the doubt instead of changing the label just to improve the score**.

This uses the already-seen test set only for descriptive error inspection. Do not optimize hyperparameters to fix these specific test examples.

## 2 — Understand the baseline

These files are the outputs you already measured:

- `results/baseline/metrics.json` — test metrics for 51 images.
- `results/baseline/confusion_matrix.png` — 14 true negatives, 6 false positives, 0 false negatives, 31 true positives.
- `results/baseline/learning_curves.png` — training and validation learning curves.
- `experiments/baseline/best_model.keras` — large local trained model; kept **off GitHub**.

The backbone of EfficientNetB0 was frozen. Only the classification head learned. This is the control experiment against which we compare fine-tuning.

## 3 — Fine-tune the baseline model

From `~/Projects/brain-mri-classification-clean` with the virtual environment activated:

```bash
python fine_tune.py --baseline-dir experiments/baseline --output-dir experiments/fine_tuned
```

What happens:

1. Load your **saved** checkpoint and its **same** training/validation image lists; no new random split.
2. Measure the original model on validation data.
3. Train the last 20 EfficientNet layers (except BatchNormalization) and the classification head with a small learning rate (`1e-5`), up to six epochs.
4. Save a new model plus learning curves and **validation** comparison, without reading the test set.

Look in `experiments/fine_tuned/`: `validation_comparison.csv`, `validation_comparison.json`, `fine_tuning_curves.png`, `best_model.keras` and `validation_predictions.csv`. These are local, untracked files. If it says an image doesn't exist, do not rerun splitting: re-download the original Kaggle dataset or restore the cached data at the old path.

**Why:** pretrained features come from ordinary photos. Unfreezing a few deeper layers lets the model adapt those features to MRI, but can overfit because there are so few images.

## 4 — Compare models fairly

```bash
cat experiments/fine_tuned/validation_comparison.csv
open experiments/fine_tuned/fine_tuning_curves.png
```

Ask: did specificity improve? Was sensitivity stable? Did balanced accuracy improve? What happened to ROC-AUC and validation loss? Prefer several metrics over accuracy alone, and be skeptical of one-image differences on only 51 validation images.

The baseline checkpoint was already chosen using validation loss; later validation comparisons are therefore exploratory and may be optimistic. Do **not** select the model using `test_predictions.csv` or repeat test evaluation after each tweak.

## 5 — What about an independent final test?

There is **no unused independent test set** in the current dataset: you have already examined the original held-out 51 test images. Do not present repeated testing on them as independent evidence of improvement. The best next step is a separately sourced cohort with documented patient-level identifiers, usable imaging and compatible labels. Before using it, confirm the license, MRI modality and class meaning. The current 2D JPG classification pipeline may need changes for DICOM/NIfTI scans or different label definitions.

Until then, report the baseline test result and fine-tuning results separately, with the limitations clearly stated.

## 6 — Publish only the small reproducible repository

The clean repository includes source, README, tests and aggregate baseline plots. Do **not** upload `experiments/`, `outputs/`, `data/`, `.venv/`, raw MRI scans or pretrained checkpoint files.

From the clean project folder:

```bash
gh auth status
git init -b main
git add .
git status --short
git commit -m "Add reproducible MRI classification baseline"
gh repo create teenanandwani/brain-mri-classification --public --source=. --remote=origin --push
```

Check the `git status --short` list BEFORE committing. Nothing with `data/`, `experiments/`, `outputs/`, `.venv/`, `.keras` or `/Users/` paths should be staged. If a repository with this name already exists, stop and ask before trying to overwrite it.

## Quick explanations for interviews

- **Transfer learning:** reuse general image patterns from ImageNet; train a new head on MRI labels.
- **Fine-tuning:** update a small part of the pretrained model with a low learning rate.
- **Sensitivity:** among labeled tumor images, how many are caught.
- **Specificity:** among labeled no-tumor images, how many are correctly rejected.
- **Data leakage:** different images or slices from the same patient could occur in both development and test groups, inflating performance.
- **Why retain baseline:** a controlled experiment must preserve the original result, not retroactively replace it.
