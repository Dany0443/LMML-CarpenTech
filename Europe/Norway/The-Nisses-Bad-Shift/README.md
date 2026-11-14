# The Nisse's Bad Shift

## Task

Find incorrect handwritten-digit labels, repair them, and retrain a classifier. Each bad shift consistently swaps two digits. The platform scores the repairs and requires F1 of at least 0.93.

## Approach

We train an RBF support vector classifier on the supplied practice images and true labels. Pixel values are divided by 16. Its digit probabilities guide a sequence model that considers normal labelling and all 45 possible digit swaps in sample-ID order. Dynamic programming chooses the sequence with the highest total log likelihood, charging a penalty of 14 when the swap state changes. We then retrain the classifier on the practice truth and repaired test labels.

## Why it works

A single unclear image can make a classifier disagree with a correct label. A bad shift produces many related disagreements in both directions. The transition penalty favours a consistent swap over a stretch of samples, while retaining the original labels elsewhere. The method learns which pair is swapped without hardcoding a pair or shift boundary.

## Run it

Use Python 3.11. From this folder, run:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Submit `test-output.csv` and `model.joblib`. The model includes pixel scaling and accepts arrays with the 64 pixel columns in numerical order. Running the script regenerates both submission files and the saved reports.

## Evidence and limits

Five-fold image predictions achieve 98.91% practice accuracy. The sequence repair recovers all 36 known label errors with no false repairs, giving practice repair F1 of 1.00. Parameters were explored on this practice set, so this is development evidence rather than an untouched benchmark.

The method proposes 30 test repairs, swapping 3 and 8 between the first changed ID `IMG_1360` and the last `IMG_1496`. These IDs identify changed samples, not proven shift boundaries. Penalties of 6, 10, 14, 20, and 30 give identical test repairs. The final classifier is trained on all 1,797 images using the supplied practice truth and proposed test corrections.

See [metrics](results/metrics.json), [practice audit](results/practice_audit.csv), [test audit](results/test_audit.csv), and [stability checks](results/stability.csv). Test truth and the platform flag are unavailable locally. The method assumes sample-ID order reflects shift order and that each bad shift swaps one pair consistently; short shifts and ambiguous boundary images may remain difficult.
