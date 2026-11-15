from __future__ import annotations
import argparse
import sys
from pathlib import Path
import model_utils as mu

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='routing_config.json', help='filename only, no path')
    parser.add_argument('--output', default='answers.json', help='filename only, no path')
    parser.add_argument('--fact-bank', default=str(mu.FACT_BANK_PATH), help='path to the question battery (defaults to dataset/fact_bank.json)')
    args = parser.parse_args()
    config_path = Path(args.config)
    if not config_path.exists():
        print(f'routing config not found: {config_path}', file=sys.stderr)
        return 2
    fact_bank_path = Path(args.fact_bank)
    if not fact_bank_path.exists():
        print(f'fact bank not found: {fact_bank_path}', file=sys.stderr)
        return 2
    fact_bank = mu.load_fact_bank(fact_bank_path)
    config = mu.load_json(config_path)
    try:
        result = mu.run_config(config, fact_bank)
    except ValueError as exc:
        print(f'invalid routing config: {exc}', file=sys.stderr)
        return 2
    answers = [{'id': qid, 'answer': info['answer']} for qid, info in result['per_question'].items()]
    mu.save_json(answers, Path(args.output))
    n_pass = sum((1 for info in result['per_question'].values() if info['pass']))
    print(f'wrote {len(answers)} answers to {args.output}')
    print(f"quality: {n_pass}/{len(answers)} questions cleared their bar -> {('PASS' if result['quality_gate'] else 'FAIL')}")
    print(f"tokens: {result['total_tokens']} / budget {mu.TOKEN_BUDGET} -> {('PASS' if result['budget_gate'] else 'FAIL')}")
    return 0
if __name__ == '__main__':
    sys.exit(main())
