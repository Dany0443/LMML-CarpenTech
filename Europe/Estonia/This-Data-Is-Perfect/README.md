# This Data Is Perfect!

## Task

Audit an e-Residency fraud model that reports 93.42% accuracy. The task asks us to check three possible causes of an inflated score, identify the real problem, and report a fairer estimate. Our result is **`SIGMOID_77`**.

## Approach

We checked the scaler, the fixed 80/20 split, and the way applications from the same person were split. We compared each change separately across 200 random seeds. For the final estimate, we grouped applications by date of birth and citizenship, then used leave-one-person-out cross-validation with the scaler inside the model pipeline.

## Why it works

The dataset has 1,971 applications from 900 people, and 81.7% of rows have a near-duplicate from the same person. With a random row split and one-nearest-neighbor model, a test application can have its duplicate in the training set. Holding out each person's applications together prevents that leakage.

The other two choices do not explain the high score. Fixing the scaler barely changes the 200-seed average, from 94.53% to 94.49%. Seed 1137 is below the average, and row-level cross-validation increases the score. Grouped splits lower the average to 76.77%; leave-one-person-out gives **1,511 correct out of 1,971, or 76.66%**. Rounded to a whole percent, that gives `SIGMOID_77`.

## Run it

Install the pinned packages with Python 3.10 to 3.12, then run from this folder:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Run `python project_pipeline.py` to reproduce the original 93.42% score.

## Evidence and limits

The [final result](evidence/final_result.txt) records the score and flag. The [run log](evidence/run_log.txt) includes the comparisons and checks. Per-seed results are in [seed_sweep.csv](evidence/seed_sweep.csv), and [variant_summary.csv](evidence/variant_summary.csv) compares their averages. The [out-of-fold predictions](evidence/oof_predictions.csv) and [dataset with person groups](evidence/applications_with_person_id.csv) let reviewers inspect individual rows.

The result describes this supplied dataset and validation setup; it does not establish future real-world fraud performance. Date of birth and citizenship are used only to group records, not as model features. If those fields do not identify people consistently, the grouping can join unrelated applications or separate one person's records.
