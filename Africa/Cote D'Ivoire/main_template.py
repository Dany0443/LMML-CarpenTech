from __future__ import annotations

import argparse
import sys
from pathlib import Path

import model_utils as mu


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="test-input.json", help="filename only, no path")
    parser.add_argument("--output", default="predictions.json", help="filename only, no path")
    parser.add_argument(
        "--model",
        default=str(mu.RECOVERED_MODEL_PATH),
        help="path to the model checkpoint to run (defaults to checkpoints/recovered_model.joblib)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"input file not found: {input_path}", file=sys.stderr)
        return 2

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"model checkpoint not found: {model_path}", file=sys.stderr)
        return 2

    vectorizer = mu.load_vectorizer()
    model = mu.load_model(model_path)

    records = mu.load_records(input_path)
    ids = [r["id"] for r in records]
    texts = [r["text"] for r in records]

    labels = mu.predict_labels(model, vectorizer, texts)
    predictions = [{"id": rid, "label": label} for rid, label in zip(ids, labels)]
    mu.save_records(predictions, Path(args.output))
    print(f"wrote {len(predictions)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
