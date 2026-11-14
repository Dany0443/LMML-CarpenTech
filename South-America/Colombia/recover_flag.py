import hashlib
import json
import unicodedata
from pathlib import Path

DATA_PATH = Path('/mnt/user-data/uploads/1794671621125_reviews.json')
EDITED_COPIES = [593, 700, 546]


def is_suspicious(char):
    name = unicodedata.name(char, '')
    return unicodedata.category(char) == 'Cf' or name.startswith(('CYRILLIC', 'GREEK'))


def anomalous_indices(samples):
    return sorted(
        s['sample_index']
        for s in samples
        if any(is_suspicious(c) for c in s['review_text'] + s['summary'])
    )


def archive_mask(indices):
    key = ','.join(str(i) for i in sorted(indices)).encode('utf-8')
    return hashlib.sha256(key).digest()


def decode_tag(tag_hex, mask):
    raw = bytes.fromhex(tag_hex)
    return bytes(a ^ b for a, b in zip(raw, mask)).rstrip(b'\x00')


def find_message(records, indices):
    mask = archive_mask(indices)
    for record in records:
        decoded = decode_tag(record['archive_tag'], mask)
        if b'SIGMOID' in decoded:
            return decoded
    return None


def main():
    data = json.loads(DATA_PATH.read_text(encoding='utf-8'))
    records = data['trusted_reference'] + data['samples']
    samples = data['samples']

    anomalies = anomalous_indices(samples)
    print('character anomalies:', anomalies)

    candidates = {
        'character anomalies': anomalies,
        'edited copies': EDITED_COPIES,
        'anomalies + edited copies': sorted(set(anomalies) | set(EDITED_COPIES)),
    }
    for label, indices in candidates.items():
        message = find_message(records, indices)
        print(f'{label}: {message if message else "no SIGMOID message found"}')


main()
