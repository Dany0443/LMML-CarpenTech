import csv
from itertools import product
import json
from pathlib import Path

import model_utils as mu

ROOT = Path(__file__).resolve().parent
bank = mu.load_fact_bank()
config = mu.load_json(ROOT / 'routing_config.json')
naive = mu.load_json(ROOT / 'dataset/example_naive_config.json')
checks = mu.run_config(config, bank)
baseline = mu.run_config(naive, bank)
expected_answers = [{'id': qid, 'answer': entry['answer']} for qid, entry in checks['per_question'].items()]
assert mu.load_json(ROOT / 'answers.json') == expected_answers
choices = []
minimum_total = 0
for qid, question in bank.items():
    passing_costs = []
    for tier, style, fewshot, context in product(mu.TIERS, mu.STYLES, (False, True), (False, True)):
        call = mu.build_call(question, tier, style, fewshot, context)
        passed = mu.question_passes(question['slots'], call['answer'])
        choices.append({'question': qid, 'tier': tier, 'style': style, 'fewshot': fewshot,
                        'context': context, 'tokens': call['tokens'], 'pass': passed})
        if passed:
            passing_costs.append(call['tokens'])
    minimum = min(passing_costs)
    assert checks['per_question'][qid]['tokens'] == minimum
    minimum_total += minimum
assert checks['quality_gate'] and checks['budget_gate']
assert not baseline['budget_gate']
incomplete = {qid: choice.copy() for qid, choice in config.items()}
incomplete['q02']['context'] = False
assert not mu.run_config(incomplete, bank)['quality_gate']
report = {
    'questions': len(bank),
    'passing_questions': sum(entry['pass'] for entry in checks['per_question'].values()),
    'tokens': checks['total_tokens'],
    'budget': mu.TOKEN_BUDGET,
    'minimum_passing_tokens': minimum_total,
    'configurations_checked': len(choices),
    'naive_tokens': baseline['total_tokens'],
    'naive_quality_passes': baseline['quality_gate'],
    'tokens_saved': baseline['total_tokens'] - checks['total_tokens'],
    'quality_gate': checks['quality_gate'],
    'budget_gate': checks['budget_gate'],
    'per_question': checks['per_question'],
}
(ROOT / 'results').mkdir(exist_ok=True)
(ROOT / 'results/verification.json').write_text(json.dumps(report, indent=2) + '\n')
with (ROOT / 'results/routing-options.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(choices[0]))
    writer.writeheader()
    writer.writerows(choices)
print(f"Quality: {report['passing_questions']}/{report['questions']}; tokens: {report['tokens']}/{report['budget']}")
print(f"Naive tokens: {report['naive_tokens']}; minimum passing total: {minimum_total}")
