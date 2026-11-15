# Where the flag lives

Flag: `SIGMOID_PROVENANCE`

## Location in the artifact

`model.safetensors` is a standard safetensors file: an 8-byte little-endian header length, a 224-byte JSON header, then a raw data block of 33,164 bytes. The header lists three tensors (`classifier.bias`, `classifier.weight`, `embedding.weight`) and no `__metadata__` block. The data block is fully covered by those three tensors, so nothing is hidden between or after them.

The payload is in `embedding.weight`, shape 256 x 32, float32. The model only ever reads rows 0 to 127, because the input is 7-bit ASCII. Rows 128 to 255 are never indexed by any valid input, so no prediction can depend on them. Row 128 is the only row whose values are not ordinary floats: its first 18 dimensions (0 to 17) are exact bfloat16 values, with the low 16 bits of each float32 equal to zero. Every other value in the tensor has random low bits, and no other row or tensor has this property.

## How it is encoded

Each of those 18 values has sign bit 1 and exponent 0x7A, so the value is `-(32 + k) / 1024` where `k` is in 0 to 31. The integer `32 + k` is the stored symbol. The flag character is recovered as `chr(64 + ((64 - (32 + k)) % 64))`, which maps the symbols onto the uppercase range and the underscore (value 31). Decoding the 18 symbols in order gives `SIGMOID_PROVENANCE`.

## Extraction code

The script in `audit_model.py` contains the extraction (`extract_payload`) and the classifier evaluation (`evaluate`). The payload step reads `embedding.weight` as uint32, takes bits 16 to 22 of row 128 (the 7-bit bfloat16 mantissa), drops the two always-zero low bits, adds 32, and maps the result to a character.

Run:

    python3 audit_model.py model.safetensors corpus.csv

Output:

    payload: SIGMOID_PROVENANCE
    accuracy: 0.9542 over 2400 records
