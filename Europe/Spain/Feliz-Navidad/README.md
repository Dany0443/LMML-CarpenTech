# ¡Feliz Navidad!

## Task

Rebuild the hidden word by deciding which souvenir purchases were made at a fair price and marking them on the pilots' board.

## Approach

For each receipt, we matched the souvenir to the price guide and checked whether the total was between the minimum and maximum price multiplied by quantity. A fair purchase marks the souvenir row and pilot column. Repeated purchases mark the same cell only once. We kept the guide's row order and all 21 pilot columns, including empty columns.

## Why it works

Using decimal arithmetic keeps values on a price boundary from shifting due to floating-point rounding. Comparing the total with quantity-scaled limits also avoids rounding a per-unit price. Preserving row and column order keeps the drawn letters in their intended positions.

## Run it

Install the packages in `requirements.txt`, then run from this folder:

```bash
python solve.py
python -m unittest discover -p 'test_*.py'
```

## Evidence and limits

Of 577 receipts, 77 pass the price check and create 57 distinct marks. The saved [board image](results/board.png) shows `cRuCiO`, read visually as `CRUCIO`. The [receipt audit](results/receipt_audit.csv) records the check for each receipt. The team reports acceptance of `SIGMOID_CRUCIO`. The script rebuilds the board; it does not automatically recognise the letters.
