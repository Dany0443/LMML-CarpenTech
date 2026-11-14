import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


def extract_address_signature(raw_address: str) -> tuple[str, str, tuple[str, ...]] | None:
    postal_match = re.search(r"([A-Z]{2,4})\s+(\d{5})", raw_address)
    if not postal_match:
        return None

    state, postal_code = postal_match.groups()
    digits = tuple(sorted(re.findall(r"\b\d+\b", raw_address)))
    if not digits:
        return None

    return state, postal_code, digits


def resolve_entities(input_path: Path) -> list[list[str]]:
    with input_path.open("r", encoding="utf-8") as file_handle:
        csv_reader = csv.DictReader(file_handle)
        record_groups = defaultdict(list)
        for row in csv_reader:
            signature = extract_address_signature(row["address"])
            if signature is None:
                continue
            record_groups[signature].append(row["record_id"])

    matched_pairs = []
    for cluster_ids in record_groups.values():
        cluster_size = len(cluster_ids)
        if cluster_size > 1:
            for first_idx in range(cluster_size):
                for second_idx in range(first_idx + 1, cluster_size):
                    matched_pairs.append([cluster_ids[first_idx], cluster_ids[second_idx]])

    return matched_pairs


def main():
    base_dir = Path.cwd()
    input_file = base_dir / "test-input.csv"
    if not input_file.is_file():
        input_file = base_dir / "data" / "train-input.csv"
        if not input_file.is_file():
            input_file = base_dir / "train-input.csv"

    output_file = base_dir / "test-output.json"

    matched_pairs = resolve_entities(input_file)
    payload = {"matches": matched_pairs}

    with output_file.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        file_handle.write("\n")


if __name__ == "__main__":
    main()
