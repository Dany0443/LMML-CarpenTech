import json
import re
import sys
from pathlib import Path


def extract_candidate_names(document_text: str) -> set[str]:
    candidate_names = set()

    name_patterns = [
        r"(?:My name is|I am|This is)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+)",
        r"My\s+(?:son|daughter|wife|husband|partner|friend|father|mother|brother|sister)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+)\s+will travel",
        r"(?:Regards|Sincerely|Best regards|Cheers),\s*\n+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+)",
        r"(?:ask for|look for|contact|traveling with|travelling with)\s+([A-ZÁÉÍÓÚÑ][a-záéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][a-záéíóúñ]+)+)",
    ]

    for pattern in name_patterns:
        for match in re.finditer(pattern, document_text):
            candidate_names.add(match.group(1).strip())

    return candidate_names


def extract_email_local_parts(document_text: str) -> set[str]:
    local_parts = set()
    for match in re.finditer(r"\b([A-Za-z0-9._%+-]+)@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", document_text):
        local_part = match.group(1).strip()
        # Only track substantial local parts (3+ chars) to prevent collisions with short words
        if len(local_part) >= 3:
            local_parts.add(local_part)
    return local_parts


def redact_document(document_text: str) -> str:
    extracted_names = extract_candidate_names(document_text)
    extracted_local_parts = extract_email_local_parts(document_text)

    # 1. IBAN Account: CR + 2 check digits + 20 digits (grouped 4-4-4-4-4-2 or glued)
    redacted = re.sub(r"\bCR\d{2}(?:\s*\d{4}){4}\s*\d{2}\b", "[ACCOUNT]", document_text)
    redacted = re.sub(r"\bCR\d{20}\b", "[ACCOUNT]", redacted)

    # 2. National ID / Cédula:
    # Dashed 9-digit Costa Rican format (\b\d-\d{4}-\d{4}\b) or glued 9 digits
    redacted = re.sub(r"\b\d-\d{4}-\d{4}\b", "[ID]", redacted)
    redacted = re.sub(
        r"((?:cédula|cedula|identity|id|citizen)(?:\s+number)?(?:\s+is)?\s*:?\s*)\b\d{9}\b",
        r"\1[ID]",
        redacted,
        flags=re.IGNORECASE,
    )

    # 3. Email addresses
    redacted = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", "[EMAIL]", redacted)

    # 4. Dates of Birth (context-aware to preserve general travel/tour dates)
    date_regex = (
        r"\b(?:\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|"
        r"(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4})\b"
    )
    redacted = re.sub(
        r"((?:date of birth|born on|birthday|birth date|DOB)(?:[^\.\n]*?is\s+|:\s*|\s+))(" + date_regex + r")",
        r"\1[DOB]",
        redacted,
        flags=re.IGNORECASE,
    )

    # 5. Phone numbers: Costa Rican (+506) or 8-digit landline/mobile
    redacted = re.sub(r"(?:\+506\s*)?[24678]\d{3}[-\s]?\d{4}\b", "[PHONE]", redacted)

    # 6. Postal Addresses: context-anchored Costa Rican addresses (Avenida/Calle + number, landmark, city)
    address_prefixes = (
        r"pickup at|pickup from|deliver to|send to|send the paperwork to|"
        r"paperwork to|written reply to|reply to|located at|home address|"
        r"residential address|address is"
    )
    redacted = re.sub(
        r"((?:" + address_prefixes + r")\s+)((?:Avenida|Calle)\s+\d+[^,\n]*,\s*[^,\n\.]+,\s*[^,\n\.]+?)(?=\s*(?:to our booking|\.|\n|$))",
        r"\1[ADDRESS]",
        redacted,
        flags=re.IGNORECASE,
    )

    # 7. Redact standalone email local parts appearing elsewhere in text
    preserved_tags = {"[NAME]", "[PHONE]", "[EMAIL]", "[ID]", "[DOB]", "[ACCOUNT]", "[ADDRESS]"}
    for local_part in sorted(extracted_local_parts, key=len, reverse=True):
        if local_part not in preserved_tags:
            redacted = re.sub(r"\b" + re.escape(local_part) + r"\b", "[NAME]", redacted, flags=re.IGNORECASE)

    # 8. Redact extracted names globally
    for candidate_name in sorted(extracted_names, key=len, reverse=True):
        if candidate_name and candidate_name not in preserved_tags:
            redacted = re.sub(r"\b" + re.escape(candidate_name) + r"\b", "[NAME]", redacted)

    return redacted


def process_documents(input_path: Path) -> dict[str, str]:
    with input_path.open("r", encoding="utf-8") as file_handle:
        data = json.load(file_handle)

    document_list = data.get("documents", [])
    output_dict = {}

    for doc in document_list:
        doc_id = doc["id"]
        text_content = doc["text"]
        output_dict[doc_id] = redact_document(text_content)

    return output_dict


def main():
    base_dir = Path.cwd()
    input_file = base_dir / "test-input.json"
    if not input_file.is_file():
        input_file = base_dir / "data" / "train-input.json"
        if not input_file.is_file():
            input_file = base_dir / "train-input.json"

    output_file = base_dir / "test-output.json"

    redacted_map = process_documents(input_file)
    payload = {"redacted_documents": redacted_map}

    with output_file.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2, ensure_ascii=False)
        file_handle.write("\n")


if __name__ == "__main__":
    main()
