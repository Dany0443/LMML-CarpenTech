# Ruta de las Identidades

## Task

Find pairs of records from three tables that refer to the same person, despite swapped, abbreviated, or misspelled names.

## Approach

The script groups records by state, five-digit postal code, and the sorted sequence of number tokens in the address. It then creates every distinct record pair within each group.

## Why it works

The task data preserves address numbers, state, and postal code for each person even when the names and address wording vary. Sorting the number tokens handles changes in their order. A group of three records contributes all three unique pairs.

## Run it

From this folder, run:

```bash
python3 main.py
```

The script reads `test-input.csv` when present, otherwise it uses `data/train-input.csv`, and writes `test-output.json`.

## Evidence and limits

The practice reference contains 12,038 matching pairs, and the solution recovered all 12,038 with no false positives or false negatives. The team reports platform acceptance. The method depends on the task's address fields being stable and unique to each person. In real data, people may share an address or have missing or changed numbers, so this signature is not universal proof of identity.
