import json
import sys
from pathlib import Path
import tiktoken

CANDIDATES = [
    "p50k_base",
    "r50k_base",
    "cl100k_base",
    "o200k_base",
]


def load_input_payload(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"Input file not found: {path}")

    with path.open("r", encoding="utf-8") as file_handle:
        try:
            payload = json.load(file_handle)
        except json.JSONDecodeError as decode_error:
            raise ValueError(f"Malformed JSON in {path}: {decode_error}") from decode_error

    if not isinstance(payload, dict):
        raise ValueError("Root JSON payload must be an object")

    calibration = payload.get("calibration")
    if not isinstance(calibration, dict):
        raise ValueError("Field 'calibration' must be an object")

    cal_text = calibration.get("text")
    if not isinstance(cal_text, str):
        raise ValueError("Field 'calibration.text' must be a string")

    cal_token_ids = calibration.get("token_ids")
    if not isinstance(cal_token_ids, list) or not all(isinstance(token_id, int) for token_id in cal_token_ids):
        raise ValueError("Field 'calibration.token_ids' must be a list of integers")

    challenge = payload.get("challenge")
    if not isinstance(challenge, dict):
        raise ValueError("Field 'challenge' must be an object")

    challenge_token_ids = challenge.get("token_ids")
    if not isinstance(challenge_token_ids, list) or not all(isinstance(token_id, int) for token_id in challenge_token_ids):
        raise ValueError("Field 'challenge.token_ids' must be a list of integers")

    return payload


def identify_tokenizer(cal_text: str, cal_token_ids: list[int]):
    available = set(tiktoken.list_encoding_names())
    search_order = [name for name in CANDIDATES if name in available]
    search_order.extend(name for name in available if name not in search_order)

    for candidate_name in search_order:
        try:
            encoding = tiktoken.get_encoding(candidate_name)
            # allowed_special='all' ensures comparison holds if calibration text includes special tokens
            encoded_ids = encoding.encode(cal_text, allowed_special="all")
            if encoded_ids == cal_token_ids:
                return candidate_name, encoding
        except Exception:
            continue

    return None, None


def solve_mystery(input_path: Path, output_path: Path) -> None:
    payload = load_input_payload(input_path)

    cal_text = payload["calibration"]["text"]
    cal_token_ids = payload["calibration"]["token_ids"]
    challenge_token_ids = payload["challenge"]["token_ids"]

    tokenizer_id, encoding = identify_tokenizer(cal_text, cal_token_ids)
    if not tokenizer_id or encoding is None:
        raise RuntimeError("No matching tokenizer could be identified from calibration tokens")

    decoded_message = encoding.decode(challenge_token_ids)

    result_payload = {
        "tokenizer_id": tokenizer_id,
        "decoded_message": decoded_message,
    }

    with output_path.open("w", encoding="utf-8") as file_handle:
        json.dump(result_payload, file_handle, indent=2, ensure_ascii=False)
        file_handle.write("\n")


def main():
    base_dir = Path.cwd()
    input_file = base_dir / "test-input.json"
    output_file = base_dir / "test-output.json"

    try:
        solve_mystery(input_file, output_file)
    except Exception as error:
        sys.stderr.write(f"Error: {error}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
