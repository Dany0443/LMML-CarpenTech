# The Weak Layer

## Task

The goal is to recover a hidden word from a decision tree fitted to the supplied data. This task is still in progress; the practice answer is not the hidden-evaluation flag.

## Approach

The current script fits a decision tree with default settings and `random_state=42`, then reads the first letter of each split feature in preorder: current split, left branch, then right branch. Leaves add no letter, and repeated features remain in the sequence.

## Why it works

The task encodes its message in the tree's split-feature sequence. Traversing the tree in a fixed preorder gives a repeatable sequence instead of relying on visual inspection or a hardcoded answer.

## Run it

Install the packages in `requirements.txt`. Put `test-input.csv` in this folder, then run:

```bash
python main.py
python -m unittest discover -p 'test_*.py'
```

The script writes `test-output.txt` in the current directory. Submit only `main.py`.

## Evidence and limits

On the practice table, the feature initials are D, E, M, and O, producing `SIGMOID_DEMO`. Four tests pass with Python 3.12 and the required package versions. This verifies the practice behavior only; it does not reveal the hidden flag or prove that the submission has been accepted.
