# Pura Vida, No PII

## Task

Remove personal information from each document while preserving its remaining text, document ID, and required JSON structure.

## Approach

The script uses regular expressions for structured values such as phone numbers and email addresses, and nearby phrases to find names, dates of birth, and addresses. It collects names and email usernames before replacing them, so later mentions can be removed too. Longer names are replaced first to avoid partial overlaps.

## Why it works

Context helps distinguish a birth date from an ordinary travel date. Keeping the document IDs and JSON structure intact preserves the expected output format. The method uses the patterns and wording found in the task data; it is not a trained entity recogniser.

## Run it

From this folder, run:

```bash
python3 main.py
```

The script uses `test-input.json` when present; otherwise it uses the practice input in `data/`. It writes `test-output.json`. The standard library is sufficient.

## Evidence and limits

The current output matches all 20 practice documents in `data/train-output.json` when compared as JSON. The team reports platform acceptance. This does not prove complete redaction on unseen documents. New formats, languages, or ambiguous text may require additional rules.
