# The Great American Token Mystery

## Task

Identify which tokenizer produced a known calibration sequence, then use it to decode a second sequence of token IDs.

## Approach

The script encodes the calibration text with each supported `tiktoken` encoding and compares the resulting IDs with the supplied sequence. An exact sequence match identifies the candidate. It then decodes the challenge IDs and writes the required JSON output.

## Why it works

Each tokenizer has its own vocabulary and segmentation rules, so the same text usually produces different token IDs across encodings. Exact comparison checks the tokenizer directly instead of guessing from whether decoded text looks readable. The candidate order resolves rare cases where a short calibration string matches more than one encoding.

## Supported encodings

The candidate pool includes `p50k_base`, `r50k_base`, `cl100k_base`, and `o200k_base`, plus registered fallbacks such as `gpt2`, `p50k_edit`, and `o200k_harmony`. It requires `tiktoken`; it does not identify unrelated tokenizer families such as SentencePiece or custom WordPiece models.

## Run it

Install the repository dependencies from the root, then run from this folder with `test-input.json` present:

```bash
python3 -m pip install -r ../../../requirements.txt
python3 main.py
```

The script reads `test-input.json` and writes `test-output.json`.

## Evidence and limits

On the supplied practice input, the script identifies `p50k_base` and decodes the message in `data/train-output.json`. The output matches the practice reference byte for byte. A trivial calibration string may not distinguish every candidate, and the script only supports encodings available through `tiktoken`.
