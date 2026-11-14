import json
import sys
from datetime import datetime

trace_path = sys.argv[1] if len(sys.argv) > 1 else "test-input.json"
with open(trace_path) as fh:
    events = json.load(fh)["events"]

runs = {}
for event in events:
    runs.setdefault(event["run_id"], []).append(event)

def answer(run_id):
    return next(e["text"] for e in runs[run_id] if e["event"] == "assistant.response")

def live_status(run_id):
    return next(e["result"]["status"] for e in runs[run_id] if e["event"] == "tool.result")

suspects = []
for run_id, steps in runs.items():
    started = datetime.fromisoformat(steps[0]["timestamp"])
    for step in steps:
        if step["event"] != "memory.result" or not step["injected_into_context"]:
            continue
        mem = step["item"]
        expired = datetime.fromisoformat(mem["valid_until"]) < started
        suspects.append((run_id, mem["source_id"], expired))

for run_id, source_id, expired in suspects:
    replay = next(r for r, steps in runs.items() if steps[0].get("replay_of") == run_id)
    print(f"run          {run_id}")
    print(f"live tool    {live_status(run_id)}")
    print(f"answered     {answer(run_id)}")
    print(f"memory       {source_id} (expired before the run: {expired})")
    print(f"replay w/o   {answer(replay)}")
    if expired and live_status(run_id) != "NOT_RESTORED" and answer(run_id) != answer(replay):
        with open("answer.txt", "w") as out:
            out.write(source_id + "\n")
        print(f"\nanswer.txt   {source_id}")
