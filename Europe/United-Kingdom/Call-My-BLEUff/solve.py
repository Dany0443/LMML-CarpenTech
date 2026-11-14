import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median

from bleu import bleu_from_stats, segment_stats, tokenize
from slots import expected, slots

HERE = Path(__file__).resolve().parent


def decode(seats):
    if len(seats) % 3 or any(s not in (1, 2, 3) for s in seats):
        raise ValueError('Expected complete triples of seats 1, 2, 3')
    numbers = [9 * (a - 1) + 3 * (b - 1) + c - 1
               for a, b, c in zip(seats[::3], seats[1::3], seats[2::3])]
    if any(n > 25 for n in numbers):
        raise ValueError('Base-three triple 222 is outside A-Z')
    return ''.join(chr(65 + n) for n in numbers)


def defect_score(texts, references, votes=3, allowance=1.20):
    losses, ratios = [], []
    for text, refs in zip(texts, references, strict=True):
        concepts = expected(refs, votes)
        if not concepts:
            raise ValueError('Reference card has no recognised consensus concepts')
        losses.append(len(concepts - slots(text)) / len(concepts))
        ratios.append(len(tokenize(text)) / median(len(tokenize(r)) for r in refs))
    return mean(losses) + max(0, mean(ratios) - allowance)


def write_csv(name, rows):
    with (HERE / 'results' / name).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    references, panel = defaultdict(list), defaultdict(dict)
    for row in csv.DictReader((HERE / 'data' / 'references.csv').open(encoding='utf-8')):
        references[int(row['round']), int(row['card'])].append(row['reference'])
    for row in csv.DictReader((HERE / 'data' / 'panel.csv').open(encoding='utf-8')):
        key, card = (int(row['round']), int(row['seat'])), int(row['card'])
        if card in panel[key]:
            raise ValueError('Duplicate panel card')
        panel[key][card] = row['text']
    assert set(references) == {(r, c) for r in range(1, 25) for c in range(1, 41)}
    assert all(len(refs) == 5 for refs in references.values())
    assert set(panel) == {(r, s) for r in range(1, 25) for s in (1, 2, 3)}
    assert all(set(deck) == set(range(1, 41)) for deck in panel.values())

    # repeated regex parsing would dominate the sensitivity sweep
    ref_concepts = {key: Counter(k for t in refs for k in slots(t))
                    for key, refs in references.items()}
    deck_rows, card_rows = [], []
    for (rnd, seat), deck in sorted(panel.items()):
        stats, losses, ratios = [], [], []
        for card, text in sorted(deck.items()):
            refs = references[rnd, card]
            consensus = {k for k, n in ref_concepts[rnd, card].items() if n >= 3}
            missing = consensus - slots(text)
            loss = len(missing) / len(consensus)
            ratio = len(tokenize(text)) / median(len(tokenize(r)) for r in refs)
            stat = segment_stats(text, refs)
            stats.append(stat)
            losses.append(loss)
            ratios.append(ratio)
            card_rows.append(dict(round=rnd, seat=seat, card=card,
                                  missing_fraction=loss, length_ratio=ratio,
                                  expected='; '.join(sorted(consensus)),
                                  missing='; '.join(sorted(missing)), text=text,
                                  references=json.dumps(refs, ensure_ascii=False),
                                  segment_bleu=bleu_from_stats([stat])['bleu']))
        bleu = bleu_from_stats(stats)
        deck_rows.append(dict(round=rnd, seat=seat, bleu=bleu['bleu'],
                              brevity_penalty=bleu['brevity_penalty'],
                              omission=mean(losses), length_ratio=mean(ratios),
                              defect=defect_score(list(deck.values()),
                                                  [references[rnd, c] for c in deck])))

    choices = []
    for rnd in range(1, 25):
        group = [row for row in deck_rows if row['round'] == rnd]
        ranked = sorted(group, key=lambda row: row['defect'])
        honest, house = ranked[0], max(group, key=lambda row: row['bleu'])
        assert ranked[1]['defect'] > honest['defect'], 'Ambiguous deck ranking'
        choices.append(dict(round=rnd, honest_seat=honest['seat'],
                            house_seat=house['seat'],
                            margin=ranked[1]['defect'] - honest['defect']))
    chosen = {row['round']: row['honest_seat'] for row in choices}
    for row in deck_rows:
        row['selected'] = row['seat'] == chosen[row['round']]

    sensitivity = []
    for votes in (2, 3, 4):
        for allowance in (1.15, 1.20, 1.25, 1.30):
            seats = []
            for rnd in range(1, 25):
                errors = []
                for seat in (1, 2, 3):
                    losses = []
                    for card, text in sorted(panel[rnd, seat].items()):
                        consensus = {k for k, n in ref_concepts[rnd, card].items()
                                     if n >= votes}
                        losses.append(len(consensus - slots(text)) / len(consensus))
                    ratio = next(row['length_ratio'] for row in deck_rows
                                 if row['round'] == rnd and row['seat'] == seat)
                    errors.append(mean(losses) + max(0, ratio - allowance))
                seats.append(1 + errors.index(min(errors)))
            sensitivity.append(dict(votes=votes, allowance=allowance,
                                    seats=''.join(map(str, seats)), word=decode(seats)))

    witnesses = []
    for deck in deck_rows:
        if deck['selected']:
            continue
        cards = [row for row in card_rows if row['round'] == deck['round']
                 and row['seat'] == deck['seat']]
        criterion = 'length_ratio' if deck['length_ratio'] > 1.20 else 'missing_fraction'
        witnesses.extend(sorted(cards, key=lambda row: row[criterion], reverse=True)[:2])
    out = HERE / 'results'
    out.mkdir(exist_ok=True)
    write_csv('decks.csv', deck_rows)
    write_csv('cards.csv', card_rows)
    write_csv('witnesses.csv', witnesses)
    write_csv('choices.csv', choices)
    write_csv('sensitivity.csv', sensitivity)
    word = decode([row['honest_seat'] for row in choices])
    house_word = decode([row['house_seat'] for row in choices])
    summary = dict(candidate_flag='SIGMOID_' + word, house_decoy='SIGMOID_' + house_word,
                   seats=[row['honest_seat'] for row in choices],
                   selected_max_defect=max(row['defect'] for row in deck_rows if row['selected']),
                   other_min_defect=min(row['defect'] for row in deck_rows if not row['selected']),
                   minimum_round_margin=min(row['margin'] for row in choices),
                   house_agreement=sum(row['house_seat'] == row['honest_seat'] for row in choices),
                   sensitivity_stable=all(row['word'] == word for row in sensitivity),
                   input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((HERE / 'data').glob('*.csv'))})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
