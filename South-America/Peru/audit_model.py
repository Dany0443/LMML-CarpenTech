import csv
import json
import struct
import sys

import numpy as np

REGISTERS = ["antisuyu", "chinchaysuyu", "qullasuyu"]
HIDDEN_ROW = 128
PAYLOAD_DIMS = 18


def load_tensors(path):
    with open(path, "rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
        data = f.read()
    base = 8 + header_len
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        raw = data[start:end]
        tensors[name] = np.frombuffer(raw, dtype="<f4").reshape(meta["shape"]).copy()
    return tensors


def extract_payload(embedding):
    bits = embedding.view("<u4")[HIDDEN_ROW, :PAYLOAD_DIMS]
    mantissa_idx = ((bits >> 16) & 0x7F) >> 2
    codes = (32 - mantissa_idx) % 64
    return "".join(chr(64 + c) for c in codes)


def evaluate(tensors, corpus_path):
    emb = tensors["embedding.weight"]
    W = tensors["classifier.weight"]
    b = tensors["classifier.bias"]
    with open(corpus_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    correct = 0
    for row in rows:
        ids = np.frombuffer(row["text"].encode("ascii"), dtype=np.uint8)
        pooled = emb[ids].mean(axis=0)
        pred = REGISTERS[int(np.argmax(W @ pooled + b))]
        correct += pred == row["register"]
    return correct / len(rows), len(rows)


if __name__ == "__main__":
    model_path, corpus_path = sys.argv[1], sys.argv[2]
    tensors = load_tensors(model_path)
    print("payload:", extract_payload(tensors["embedding.weight"]))
    acc, total = evaluate(tensors, corpus_path)
    print(f"accuracy: {acc:.4f} over {total} records")
