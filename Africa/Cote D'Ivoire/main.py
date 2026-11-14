import argparse
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

import model_utils as mu


def build_model():
    feats = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
    ])
    return Pipeline([("feats", feats), ("clf", LogisticRegression(C=20, max_iter=2000))])


def load_training_rows():
    folder = mu.HERE / "dataset"
    old_rows = mu.load_records(folder / "old_task_sample.json")
    texts = {r["id"]: r["text"] for r in mu.load_records(folder / "train-input.json")}
    public_rows = [
        {"text": texts[r["id"]], "label": r["label"]}
        for r in mu.load_records(folder / "train-output.json")
    ]
    return old_rows + public_rows


parser = argparse.ArgumentParser()
parser.add_argument("--input", default="test-input.json", help="filename only, no path")
parser.add_argument("--output", default="predictions.json", help="filename only, no path")
args = parser.parse_args()

input_path = Path(args.input)
if not input_path.exists():
    print(f"input file not found: {input_path}", file=sys.stderr)
    sys.exit(2)

rows = load_training_rows()
model = build_model()
model.fit([r["text"] for r in rows], [r["label"] for r in rows])
mu.save_model(model, mu.RECOVERED_MODEL_PATH)

records = mu.load_records(input_path)
labels = model.predict([r["text"] for r in records])
predictions = [{"id": r["id"], "label": str(lab)} for r, lab in zip(records, labels)]
mu.save_records(predictions, Path(args.output))
print(f"wrote {len(predictions)} predictions to {args.output}")
