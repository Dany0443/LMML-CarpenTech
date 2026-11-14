# Call My BLEUff

## Task

Choose the honest participant in each of 24 rounds. Each deck has 40 venue descriptions, and the honest participant describes every venue well.

## Approach

BLEU rewards matching word sequences, even when prose is padded or misses useful facts. We extracted details that appeared in at least three of each card's five references. For each deck, we averaged the fraction of expected details missing and added a penalty when average description length exceeded 1.20 times the reference median. The lowest penalty identifies the candidate honest deck.

## Why it works

The omission score checks whether descriptions retain the venue facts that recur across references. The length penalty catches padding that can still preserve those words and score well under BLEU. The rules are specific to the supplied venue corpus; they are not a general language-understanding system.

## Run it

From this folder, run:

```bash
python3 main.py
python3 -m unittest discover -p 'test_*.py'
```

## Evidence and limits

The selected seats stayed the same across 12 tested parameter combinations. Converting seats to base-3 digits and reading three rounds per letter gives the literal string `FRINDLOW`. The team reports that this flag was accepted. See [deck scores](results/decks.csv), [seat choices](results/choices.csv), and the [comparison figure](results/deck_comparison.png). This is a stability check on the supplied corpus, not a held-out NLP accuracy measurement. The scorer lowercases text and separates Unicode words and punctuation.
