# The Canal Knows Too Much

## Task

Audit an 80-question benchmark for answers already present in the model's training corpus, then report accuracy on the questions left after removing those contaminated items.

## Approach

We compared each question and answer with 400 English and Spanish records. We counted a match only when the record supported the same fact about the same subject. The audit accounts for translations, equivalent wording, and number or unit formats, while keeping related but different facts in the benchmark.

## Why it works

A shared number alone does not establish contamination. Requiring the same subject and requested attribute avoids excluding questions just because a similar measurement appears elsewhere. The audit lists each removed question beside its supporting corpus record so the decision can be checked.

## Run it

From this folder, validate the saved report with:

```bash
python3 open_seal.py report.json
```

The audit evidence and report are saved in `canalqa_audit.json`, `audit_seal.json`, and `report.json`. The detailed question-by-question reasoning is in [writeup.md](writeup.md).

## Evidence and limits

The original score is 59/80, or 73.75%. The audit excludes 21 contaminated questions, all originally correct, leaving 38/59 or 64.41% on the remaining questions. The seal validator accepts the saved report and displays `SIGMOID_DECONTAMINATION`. The conclusion depends on the supplied corpus and the documented equivalence judgments; unprovided training records are outside this audit.
