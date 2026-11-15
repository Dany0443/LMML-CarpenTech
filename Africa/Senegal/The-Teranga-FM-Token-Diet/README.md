# The Teranga FM Token Diet

## Task

Answer 20 questions while covering every required fact and using at most 1,200 tokens in total. Both prompts and completions count. Each question can use a cheap or large model, terse or verbose wording, an optional example block and optional context.

## Approach

We try all 16 combinations for each question using the supplied deterministic simulator. We discard choices that miss a required fact and keep the cheapest passing choice. The resulting routing is saved in `routing_config.json`, and `main.py` regenerates `answers.json` from it.

Every selected answer is terse and uses no example block. Eight questions use the cheap model alone, six use the cheap model with context and six use the large model. Short context can restore facts the cheap model would otherwise miss. When context is too long, the large model can cost fewer tokens overall.

## Why it works

Each question has an independent token cost and quality check. There are no shared model limits or interactions between calls. Therefore, adding up the cheapest passing choice for every question gives the minimum possible total under the simulator's rules. We reduce unnecessary wording without dropping required facts.

## Run it

Python 3 and the standard library are enough. From this task folder:

```bash
python solve_routing.py
python main.py
python verify.py
```

The first command rebuilds the routing. The second writes the answers using the default filenames. The verifier checks the saved answers, evaluates every routing option and confirms that the chosen cost is minimal. It also checks that removing needed context fails the quality gate.

Keep `main.py`, `routing_config.json`, `model_utils.py` and the `dataset/` folder together. No API keys, model downloads or internet access are needed.

## Evidence and limits

| Configuration | Questions passing | Tokens | Within budget |
| --- | ---: | ---: | --- |
| Supplied naive routing | 20/20 | 3,626 | No |
| Our routing | 20/20 | 1,160 | Yes |

The routing saves 2,466 tokens and leaves 40 tokens below the limit. `results/verification.json` records the checks and each answer's cost. `results/routing-options.csv` lists all 320 choices examined across the 20 questions. The regenerated routing and answers match the supplied submission archive.

This is an exact solution for a fixed simulated gateway, not a benchmark of real language models. Tokens are counted with the supplied word-and-punctuation rule, and quality is determined by substring coverage of fact slots. The explanation follows those challenge rules rather than claiming that the answers have been independently fact-checked. The evaluator script itself was not included in the submission archive; verification uses the shared scoring functions in `model_utils.py`.
