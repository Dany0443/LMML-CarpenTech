import json
from main import ask_model, expected, MODEL
from llama_cpp import Llama

llm = Llama(model_path=str(MODEL), n_gpu_layers=-1, n_ctx=2048, verbose=False)
lines = open("test-input.txt", encoding="utf-8-sig").read().splitlines()
bad = 0
for n, line in enumerate(lines, 1):
    want = expected(line)
    try:
        got = ask_model(llm, line)
    except Exception as e:
        got = f"ERR {e}"
    if got != want:
        bad += 1
        print(f"--- line {n}\n in:   {line}\n model: {got}\n rule:  {want}")
print(f"{bad}/{len(lines)} lines differ from the rules")
