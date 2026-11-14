# Always Valid, No Excuses

## Task

Turn OCR-style lines of pipe-separated `key: value` fields into JSON objects. Empty values become `null`, exact `true` and `false` values become booleans, whole numbers become integers, and other values stay strings.

## Approach

The program asks a local Qwen model to format each line as JSON. It checks the model's response against the values produced by the field rules. If the whole-line response is wrong, it retries one field at a time. A deterministic fallback returns the rule-based value if the model still gives an incorrect answer.

## Why it works

The model helps with formatting, while exact validation prevents its guesses from changing the parsed values. Splitting each field at its first colon preserves colons inside the value, and JSON serialization handles quotes and backslashes.

## Run it

Install a compatible `llama-cpp-python` build and place `Qwen3-1.7B-Q8_0.gguf` at `models/Qwen3-1.7B-Q8_0.gguf` in this folder. Then run:

```bash
python3 main.py
```

The script reads `test-input.txt` and writes `test-output.txt` in this folder. The model file is excluded from Git and must be supplied separately.

## Evidence and limits

The repository includes sample input and output files for checking the JSON format. The saved training output contains 13 records. No separate evaluation score is recorded in this README. The field rules only convert empty strings, exact lowercase booleans, and signed whole numbers; other values remain strings. Results depend on having the local model and a working `llama-cpp-python` installation.
