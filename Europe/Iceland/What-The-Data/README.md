# What The Data?

## Task

Turn the damaged station export into a clean table with one Celsius reading per ID, sorted by ID and formatted exactly as required.

## Approach

The script skips comment and blank lines, reads the semicolon-separated table, fixes decimal commas, repairs reversible UTF-8 decoding mistakes, and converts Fahrenheit with `(F - 32) * 5 / 9`. It normalises readings before removing duplicate IDs, sorts numerically, then writes Celsius to one decimal place.

## Why it works

Cleaning names, units, and numbers before deduplication allows retransmissions with different raw formatting to collapse to one reading. The encoding repair reverses common Latin-1 and Windows-1252 mistakes without a station lookup table. The output formatting follows the requested schema exactly.

## Run it

From this folder, copy the practice input into the expected working filename and run the cleaner:

```bash
cp data/train-input.csv test-input.csv
python3 main.py
cmp test-output.csv data/train-output.csv
python3 -m unittest discover -p 'test_*.py'
```

Submit only `main.py`. It reads `test-input.csv` and writes `test-output.csv` in the current directory.

## Evidence and limits

The practice output matches all 30 reference rows byte for byte. Four tests cover unseen names, repeated encoding damage, extra duplicates, and longer preambles. The team reports platform acceptance. The cleaner assumes duplicate IDs refer to the same reading. Characters already replaced by a replacement symbol cannot be recovered reliably.
