# Just a siblings' thing

## Task

Choose which of three supplied models should detect catastrophic threats. All models have the same accuracy, but missing a real threat is more costly than a false alarm.

## Approach

We loaded the supplied Ares, Athena, and Apollo weights without retraining. We joined labels by encounter ID, used the feature order in `model.py`, converted each model's logits to probabilities with sigmoid, and applied the required 0.50 threshold.

## Why it works

Accuracy counts false alarms and missed threats equally, so it hides the difference that matters here. All three models make 50 errors, but Ares misses the fewest catastrophic encounters. If a miss costs more than a false alarm, Ares has the lowest total error cost.

## Run it

Install the packages listed in `requirements.txt`, including PyTorch 2.6.0. From this folder, run:

```bash
python main.py
python -m unittest discover -p 'test_*.py'
```

## Evidence and limits

There are 100 catastrophic and 900 normal encounters. Ares has 95 true positives and five false negatives; Athena has 75 and 25; Apollo has 55 and 45. All models have 95% accuracy. The [metrics](results/metrics.csv) and [predictions](results/predictions.csv) contain the comparison, and the metrics were checked independently with scikit-learn. The team reports acceptance of `SIGMOID_ARES`. The recommendation assumes that a missed threat is more costly than a false alarm; the precise cost was not supplied.
