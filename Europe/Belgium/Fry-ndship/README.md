# Fry-ndship

## Task

Predict the edge length of potato fields from the measurements in the supplied table, then inspect the prediction errors for the hidden pattern.

## Approach

We converted area from hectares to square metres and derived the short side with `10000 * area_ha / long_side_m`. We trained a linear regression model using both side lengths and declared ditch length. We then plotted observed edge length minus predicted edge length against the prediction.

## Why it works

For a rectangular field, perimeter depends on both side lengths: `2 * (long_side + short_side)`. The derived short side adds that geometric information to the model. The residual plot exposes a pattern that a summary score such as R² does not show. We kept on-site hedge measurements out of the prediction features because they would not be available for an unsurveyed field.

## Run it

Install NumPy, pandas, scikit-learn, and Matplotlib. From this folder, run:

```bash
python main.py
```

## Evidence and limits

An 80/20 split with `random_state=42` gave test R² of 0.999984 and MAE of about 0.786 metres. The residual figure reveals `SUPERVISION`; the pattern also appears in rows excluded from fitting and disappears when residuals are shuffled. The team reports acceptance of `SIGMOID_SUPERVISION`. See [residual evidence](figures/survey_evidence.png) and [model comparison](figures/model_comparison.png). The residual pattern was found during exploration, so this is supporting evidence, not an untouched benchmark.
