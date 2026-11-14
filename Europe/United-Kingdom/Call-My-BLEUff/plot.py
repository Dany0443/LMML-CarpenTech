import csv
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
with (HERE / 'results' / 'decks.csv').open() as stream:
    decks = list(csv.DictReader(stream))

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
for seat, color in [(1, '#0072B2'), (2, '#D55E00'), (3, '#009E73')]:
    seat_decks = [deck for deck in decks if int(deck['seat']) == seat]
    rounds = [int(deck['round']) for deck in seat_decks]
    axes[0].scatter(rounds, [float(deck['bleu']) for deck in seat_decks],
                    label=f'Seat {seat}', color=color, s=35)
    axes[1].scatter(rounds, [float(deck['defect']) for deck in seat_decks],
                    color=color, s=35)
chosen = [deck for deck in decks if deck['selected'] == 'True']
for ax, metric in zip(axes, ['bleu', 'defect']):
    ax.scatter([int(deck['round']) for deck in chosen],
               [float(deck[metric]) for deck in chosen],
               s=100, facecolors='none', edgecolors='black', linewidths=1.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=.2)
axes[0].set_ylabel('House BLEU (higher wins)')
axes[0].legend(frameon=False, ncol=3)
axes[0].set_title('Circles mark the seat selected by the proposed check')
axes[1].set_ylabel('Defect score (lower wins)')
axes[1].axhspan(.021905, .032143, color='#bbbbbb', alpha=.3)
axes[1].set_xlabel('Round')
axes[1].set_xticks(range(1, 25))
fig.tight_layout()
fig.savefig(HERE / 'results' / 'deck_comparison.png', dpi=160)
