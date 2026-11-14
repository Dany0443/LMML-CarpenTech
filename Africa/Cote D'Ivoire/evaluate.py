import json
import sys

import numpy as np
from sklearn.model_selection import StratifiedKFold

import model_utils as mu

sys.argv = ["main.py", "--input", "dataset/train-input.json", "--output", "/tmp/unused.json"]
src = open("main.py").read().split("parser = argparse")[0]
exec(src)

rows = load_training_rows()
texts = np.array([r["text"] for r in rows])
labels = np.array([r["label"] for r in rows])
print(f"pooled rows: {len(rows)} ({sum(l in mu.OLD_LABELS for l in labels)} old, "
      f"{sum(l in mu.NEW_LABELS for l in labels)} new)")

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
hits = {"old": [], "new": []}
for fit_idx, test_idx in folds.split(texts, labels):
    model = build_model()
    model.fit(texts[fit_idx], labels[fit_idx])
    gold = {str(i): labels[i] for i in test_idx}
    pred = dict(zip(gold, model.predict(texts[test_idx])))
    scores = mu.accuracy_by_task(gold, pred)
    for task in hits:
        hits[task].append(scores[task])
print("5-fold cv on pooled data (old / new):")
for task in hits:
    print(f"  {task}: mean {np.mean(hits[task]):.3f}  min {np.min(hits[task]):.3f}")

old_sample = mu.load_records("dataset/old_task_sample.json")
public = load_training_rows()[len(old_sample):]
new_only = [r for r in public if r["label"] in mu.NEW_LABELS]
old_holdout = [r for r in public if r["label"] in mu.OLD_LABELS]
model = build_model()
train = old_sample + new_only
model.fit([r["text"] for r in train], [r["label"] for r in train])
gold = {str(i): r["label"] for i, r in enumerate(old_holdout)}
pred = dict(zip(gold, model.predict([r["text"] for r in old_holdout])))
print(f"replay only ({len(old_sample)} old rows + {len(new_only)} new rows) on {len(old_holdout)} unseen old rows: "
      f"{mu.accuracy_by_task(gold, pred)['old']:.3f}")
