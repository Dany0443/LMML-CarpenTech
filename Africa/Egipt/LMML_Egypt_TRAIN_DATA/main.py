import json
import re
from pathlib import Path

from llama_cpp import Llama

HERE = Path(__file__).resolve().parent
MODEL = HERE / "models" / "Qwen3-1.7B-Q8_0.gguf"
KEY_RE = re.compile(r"^[A-Za-z_]+$")
INT_RE = re.compile(r"^-?\d+$")
MAX_TRIES = 2

SYSTEM = (
    "You convert OCR fields into strict JSON. Each input line is key: value. "
    "Output one JSON object with exactly the given keys. Trim spaces. Keep every character "
    "of each value, including spaces, colons, quotes and backslashes. "
    "Empty value -> null. Exactly true or false -> boolean. Whole number -> number. "
    "Anything else -> string with \" and \\ escaped. Output only the JSON object, nothing else."
)

LINE_EXAMPLES = [
    ('king: Menes\ntitle: ruled with "the Aten"\nalive: true',
     '{"king":"Menes","title":"ruled with \\"the Aten\\"","alive":true}'),
    ("ratio: 3:1\nrank:\nage: -3",
     '{"ratio":"3:1","rank":null,"age":-3}'),
]

FIELD_EXAMPLES = [
    ("king: Menes", '{"king":"Menes"}'),
    ('title: ruled with "the Aten"', '{"title":"ruled with \\"the Aten\\""}'),
    ("alive :true", '{"alive":true}'),
    ("year: -1264", '{"year":-1264}'),
    ("epithet:", '{"epithet":null}'),
    ("ratio: 3:1", '{"ratio":"3:1"}'),
    ("cost: $200", '{"cost":"$200"}'),
]


def canon(obj):
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def expected_value(raw):
    if raw == "":
        return None
    if raw == "true":
        return True
    if raw == "false":
        return False
    if INT_RE.match(raw):
        return int(raw)
    return raw


def fields(line):
    out = []
    for segment in line.split("|"):
        segment = segment.strip()
        if ":" not in segment:
            continue
        key, raw = segment.split(":", 1)
        key, raw = key.strip(), raw.strip()
        if KEY_RE.match(key):
            out.append((key, raw))
    return out


def extract_object(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object")
    obj = json.loads(text[start:end + 1])
    if not isinstance(obj, dict):
        raise ValueError("not an object")
    return obj


def chat(llm, examples, body, max_tokens, error=None):
    messages = [{"role": "system", "content": SYSTEM}]
    for src, out in examples:
        messages.append({"role": "user", "content": src + " /no_think"})
        messages.append({"role": "assistant", "content": out})
    prompt = body + " /no_think"
    if error:
        prompt += f"\nPrevious answer was wrong: {error}. Output only the corrected JSON."
    out = llm.create_chat_completion(
        messages=messages + [{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=0,
    )
    return out["choices"][0]["message"]["content"]


def convert_field(llm, key, raw, stats):
    want = {key: expected_value(raw)}
    error = None
    for _ in range(MAX_TRIES):
        try:
            obj = extract_object(chat(llm, FIELD_EXAMPLES, f"{key}: {raw}", 64, error))
            if canon(obj) == canon(want):
                stats["field"] += 1
                return want
            error = (f"got {json.dumps(obj, ensure_ascii=False)}, "
                     f"expected {json.dumps(want, ensure_ascii=False)}")
        except (ValueError, json.JSONDecodeError) as exc:
            error = str(exc)
    stats["fallback"] += 1
    return want


def convert_line(llm, line, stats):
    pairs = fields(line)
    if not pairs:
        return "{}"
    want = {k: expected_value(raw) for k, raw in pairs}
    body = "\n".join(f"{k}: {raw}" for k, raw in pairs)
    try:
        obj = extract_object(chat(llm, LINE_EXAMPLES, body, 256))
        if canon(obj) == canon(want):
            stats["line"] += 1
            return canon(obj)
    except (ValueError, json.JSONDecodeError):
        pass
    record = {}
    for key, raw in pairs:
        record.update(convert_field(llm, key, raw, stats))
    return json.dumps(record, ensure_ascii=False)


def main():
    llm = Llama(model_path=str(MODEL), n_gpu_layers=-1, n_ctx=2048, verbose=False)
    src = HERE / "test-input.txt"
    dst = HERE / "test-output.txt"
    stats = {"line": 0, "field": 0, "fallback": 0}
    with src.open(encoding="utf-8-sig") as fh:
        lines = fh.read().splitlines()
    with dst.open("w", encoding="utf-8", newline="\n") as fh:
        for line in lines:
            fh.write(convert_line(llm, line, stats) + "\n")
    print(stats)


if __name__ == "__main__":
    main()