import csv
import sys


def read_pairs(path):
    pairs = set()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            a, b = row["image_a"].strip(), row["image_b"].strip()
            if a and b and a != b:
                pairs.add(tuple(sorted((a, b))))
    return pairs


def score(guess, truth):
    tp = len(guess & truth)
    fp = len(guess - truth)
    fn = len(truth - guess)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


if __name__ == "__main__":
    guess = read_pairs(sys.argv[1])
    truth = read_pairs(sys.argv[2])
    prec, rec, f1 = score(guess, truth)
    print(f"precision {prec:.3f}  recall {rec:.3f}  F1 {f1:.3f}")
    print(f"tp {len(guess & truth)}  fp {len(guess - truth)}  fn {len(truth - guess)}")
    for pair in list(guess - truth)[:10]:
        print("false positive", pair)
    for pair in list(truth - guess)[:10]:
        print("missed", pair)