# Quipu register classifier: audit

## Problem

A character-level classifier was built to tag transcribed quipu records with one of three provincial recording conventions: `chinchaysuyu`, `antisuyu` or `qullasuyu`. It has been in production for eight months. The person who trained it has left and the handover included only the model file, with no training code or notes.

This audit has two parts. The first is to confirm the model does what it claims. The second is to check whether the file carries anything its behaviour does not reveal. The second part turned up a hidden payload, described in `writeup.md`.

## Approach

The model reads each record as raw ASCII bytes, looks up a 32-dimensional vector per byte in an embedding table, averages the vectors across the record, and applies a linear layer to produce three scores. The file contains only the weights, so I reconstructed that pipeline from the tensor shapes and checked it against the labelled corpus.

For the hidden content, I looked at what the model cannot reach. Inputs are ASCII, so only embedding rows 0 to 127 are ever read. Rows 128 to 255 are unreachable, which makes them the obvious place to look. Row 128 contains 18 values with unusually regular bit patterns, which turned out to be the payload.

## Why it works

Mean pooling and a linear head are a standard setup for a model like this, and the corpus results below confirm that the reconstruction matches the original behaviour. The payload sits in rows that are never read during inference, so the classifier's outputs are unaffected by it, which is why eight months of production use did not expose it.

## Input encoding and tokenizer

There is no tokenizer in the usual sense. The model is byte-level: each input character is its ASCII code, used directly as an index into the embedding table. Records are fixed at 64 characters.

## Supported encodings

- Input: 7-bit ASCII. Bytes 0 to 127 are valid. Bytes 128 to 255 index unused embedding rows and should not be fed in.
- Record length: 64 characters. The corpus has no records of any other length.
- Alphabet in the corpus: the same inventory of sounds across all three registers, differing only in frequency.

## Installation

Requires Python 3.9 or newer.

    pip install numpy

No other dependencies are needed. The safetensors file is parsed directly, so the `safetensors` package is not required.

## Running the audit

    python3 audit_model.py 1794735524895_model.safetensors 1794735524895_corpus.csv

Expected output:

    payload: SIGMOID_PROVENANCE
    accuracy: 0.9542 over 2400 records

## Test results

No training code was provided, so there are no training logs or validation curves. The results below come from evaluating the shipped weights on the 2,400-record corpus (800 per register).

Accuracy is 95.42% overall. Per class:

| Register      | Correct | Total | Accuracy |
|---------------|---------|-------|----------|
| antisuyu      | 762     | 800   | 95.25%   |
| chinchaysuyu  | 758     | 800   | 94.75%   |
| qullasuyu     | 770     | 800   | 96.25%   |

Label order: the output units map to `antisuyu`, `chinchaysuyu` and `qullasuyu` in that order. This mapping is not stated anywhere in the file. It is the only one that gives above-chance accuracy, and the other five permutations score between 2% and 33%. Anyone deploying the model should confirm it against the original documentation.

## Known limitations

- No training code, config or notes were provided. Architecture details (mean pooling, the absence of a nonlinearity, the label order) were inferred from the tensor shapes and the corpus, not from documentation.
- The corpus is synthetic-looking: all records are exactly 64 characters, and the three registers are reconstructions rather than attested languages. Accuracy on real transcriptions may differ.
- The 95.42% figure is measured on the same corpus the model was presumably evaluated on. It is not a held-out estimate.
- The payload is not a security control. It is a deliberately hidden string, and nothing in the model's behaviour depends on it. Its presence says nothing about whether the training process was otherwise sound.
- The audit only checks the file as shipped. It cannot prove the absence of other hidden content beyond what was inspected: the header, all three tensors, and the unused embedding rows.
