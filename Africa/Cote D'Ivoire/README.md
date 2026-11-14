# The Abidjan Transit Ledger

**Category:** ML / catastrophic forgetting  **Difficulty:** Medium  **Flag format:** `SIGMOID_<WORD>`

The flag is released by the platform once both accuracy gates pass on the hidden test split. I can't see that split, so no flag is recorded here.

## Problem statement

A small text classifier routes a clerk's plain-language request to one of ten query-template labels: five for the old transactional ledger (`OLD_*`) and five for the new spatial and window-function work (`NEW_*`). After fine-tuning on the new task only, it scores old 0.00 / new 1.00. The original checkpoint is the mirror image, old 1.00 / new 0.00. The job is to submit a `main.py`, with the I/O contract of `main_template.py` unchanged, that produces a model scoring at least 0.85 on both tasks on a hidden test set within 120 seconds on CPU. Only `predictions.json` is graded.

## Approach

`main.py` ignores the two shipped checkpoints and trains one new classifier on all labelled data it has been given:

- `dataset/old_task_sample.json` (40 old-task rows), and
- `dataset/train-input.json` + `train-output.json` (250 rows, 25 per label, both tasks).

The model is a scikit-learn pipeline: word 1-2-gram TF-IDF joined with character 2-5-gram TF-IDF, feeding a logistic regression (C=20). It then predicts the hidden `test-input.json` and writes `predictions.json`. It also saves the trained model to `checkpoints/recovered_model.joblib`, which the task allows but doesn't require.

## How it works

The old model forgot because the second round of training only showed it new-task labels, so nothing kept the old decision boundaries alive. Training once on both label sets at the same time removes that failure by construction: the classifier sees all ten classes together and has no reason to drop five of them. The requests are short, templated phrases ("drop the account row for Fatou", "nearest warehouse to Bouaké"), where distinctive words and word fragments decide the label, which linear models on TF-IDF handle well. Character n-grams help with paraphrases and unseen names.

## Installation and execution

Python 3.10-3.12 with scikit-learn, numpy and joblib (the task pins 1.8.0 / 2.5.1 / 1.5.3). Locally I tested on scikit-learn 1.9.1, numpy 2.5.4 and joblib 1.6.0.

```bash
python main.py
```

With no arguments it reads `test-input.json` and writes `predictions.json` in the working directory. `--input` and `--output` still work. It needs `model_utils.py` and the `dataset/` folder next to it, as the harness provides.

To reproduce the numbers below:

```bash
python evaluate.py
```

## Training results

Everything below uses only data I had. The hidden test split was not available, so these are estimates, not the graded score.

| Check | Old | New |
|---|---|---|
| 5-fold stratified CV on the pooled 290 rows (mean) | 0.988 | 1.000 |
| 5-fold CV, worst fold | 0.939 | 1.000 |
| Old-task replay only: train on the 40 old rows plus the 125 new rows, test on 125 unseen old rows | 0.952 | - |

Both gates (0.85) clear with margin in every setting. A full run of `main.py` on a 250-row input took about 1.3 s, far under the 120 s cap. The output has exactly `id` and `label` per row, and every label is in the ten-label set.

## Known limitations

- **Not tested against the real harness.** I never had the checkpoints, `vectorizer.joblib` or the hidden split, so the ≥0.85 result is a cross-validation estimate. The hidden set may be phrased differently from the public one.
- **Not a model recovery.** `main.py` does not merge or repair the fine-tuned checkpoint. The task says any approach is acceptable, but if you want the original weights recovered (for example with replay or weight averaging), that is a different solution.
- **Trains on the public self-check set.** That set was meant for local checking. Using it for training is why the numbers above come from cross-validation and not from scoring the model on its own training rows.
- **Old-task data is thin.** Only 165 of the pooled rows are old-task. Phrasings far outside the templates (new verbs, other languages) could be misrouted. The old task is the weaker side, with a worst-fold 0.939.
- **Version drift.** I tested on scikit-learn 1.9.1, not the pinned 1.8.0. `main.py` trains at run time and loads no pickles, so the mismatch should not matter, but I haven't run it on 1.8.0.
