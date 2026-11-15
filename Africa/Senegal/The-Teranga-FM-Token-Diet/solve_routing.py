from itertools import product
import json
from pathlib import Path
import model_utils as mu
fact_bank = mu.load_fact_bank()
config, total = ({}, 0)
for qid, q in fact_bank.items():
    options = []
    for tier, style, fewshot, context in product(mu.TIERS, mu.STYLES, (False, True), (False, True)):
        call = mu.build_call(q, tier, style, fewshot, context)
        if mu.question_passes(q['slots'], call['answer']):
            options.append((call['tokens'], tier == 'large', style == 'verbose', fewshot, context, dict(tier=tier, style=style, fewshot=fewshot, context=context)))
    if not options:
        raise SystemExit(f'{qid}: no passing configuration exists')
    tokens, *_, choice = min(options, key=lambda o: o[:5])
    config[qid] = choice
    total += tokens
    print(f'{qid}: {tokens:3d} tok  {choice}')
Path('routing_config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
print('total', total, 'budget', mu.TOKEN_BUDGET)
