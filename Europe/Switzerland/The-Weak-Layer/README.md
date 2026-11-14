# The Weak Layer

We trained a decision tree with default settings and `random_state=42`, then read the first letter of each split feature in tree order, skipping the leaves. On the practice table, the features start with D, E, M, and O, giving `SIGMOID_DEMO`.

`main.py` derives the word from each fitted tree, so the practice answer is not hardcoded. All four tests pass with Python 3.12 and the required package versions.

Install the dependencies with `pip install -r requirements.txt`, then run `python main.py` from the directory containing `test-input.csv`. The program writes the flag to `test-output.txt` in that same directory.

**Submit only `main.py`.** `SIGMOID_DEMO` is the practice answer, not the hidden test flag.
