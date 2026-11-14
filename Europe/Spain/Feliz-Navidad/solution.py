import argparse
import json
from decimal import Decimal
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def build_board(purchases, guide):
    if guide.Product.duplicated().any():
        raise ValueError('Each product must have exactly one price guide entry.')
    office = guide.assign(List_no=guide.List_no.astype(int)).sort_values('List_no')
    if office.List_no.tolist() != list(range(1, 7)):
        raise ValueError('The office guide must contain list positions 1 through 6.')
    audit = purchases.merge(office, on='Product', how='left', sort=False, validate='many_to_one')
    if audit.List_no.isna().any():
        raise ValueError('Every purchased product must appear in the price guide.')
    pilots = pd.to_numeric(audit.Pilot_no, errors='raise')
    if not pilots.isin(range(1, 22)).all():
        raise ValueError('Pilot numbers must be integers from 1 through 21.')
    fair = []
    unit_prices = []
    for quantity, total, lower, upper in audit[
        ['Quantity', 'Total_paid_eur', 'Fair_min_eur', 'Fair_max_eur']
    ].itertuples(index=False, name=None):
        quantity, total, lower, upper = map(Decimal, map(str, [quantity, total, lower, upper]))
        if not all(number.is_finite() for number in [quantity, total, lower, upper]):
            raise ValueError('Quantities and prices must be finite.')
        if quantity <= 0 or lower > upper:
            raise ValueError('Quantities must be positive and price ranges must be ordered.')
        fair.append(lower * quantity <= total <= upper * quantity)
        unit_prices.append(str(total / quantity))
    audit['Pilot_no'] = pilots.astype(int)
    audit['Unit_price_eur'] = unit_prices
    audit['Fair'] = fair
    accepted = audit.loc[audit.Fair]
    checks = pd.crosstab(accepted.List_no, accepted.Pilot_no)
    board = checks.reindex(index=range(1, 7), columns=range(1, 22), fill_value=0).gt(0).astype(int)
    board.index.name = 'List_no'
    board.columns.name = 'Pilot_no'
    return board, audit


def draw_board(board, guide, destination):
    office = guide.assign(List_no=guide.List_no.astype(int)).sort_values('List_no')
    fig, ax = plt.subplots(figsize=(16, 6), layout='constrained')
    fig.patch.set_facecolor('#faf7f0')
    ax.imshow(board.to_numpy(), cmap=ListedColormap(['#faf7f0', '#9f2436']),
              vmin=0, vmax=1, interpolation='nearest', aspect='equal', origin='upper')
    ax.set_xticks(np.arange(21), labels=np.arange(1, 22), fontsize=10)
    ax.set_yticks(np.arange(6), labels=[f'{row.List_no}  {row.Product}' for row in office.itertuples()], fontsize=11)
    ax.set_xticks(np.arange(-0.5, 21, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 6, 1), minor=True)
    ax.grid(which='minor', color='#ded8cc', linewidth=0.7)
    ax.tick_params(which='both', length=0, pad=8)
    ax.set_xlabel('Pilot number', fontsize=11, labelpad=14)
    ax.set_ylabel('Office list', fontsize=11, labelpad=14)
    ax.set_title('The Christmas board', fontsize=23, fontweight='bold', loc='left', pad=35)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.savefig(destination, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def run():
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description='Rebuild the Spanish souvenir board from fair-price receipts.')
    parser.add_argument('--purchases', type=Path, default=folder / 'data/purchases.csv')
    parser.add_argument('--guide', type=Path, default=folder / 'data/office_price_guide.csv')
    parser.add_argument('--destination', type=Path, default=folder / 'results')
    args = parser.parse_args()
    purchases = pd.read_csv(args.purchases, dtype=str, encoding='utf-8-sig')
    guide = pd.read_csv(args.guide, dtype=str, encoding='utf-8-sig')
    board, audit = build_board(purchases, guide)
    args.destination.mkdir(parents=True, exist_ok=True)
    board.to_csv(args.destination / 'board.csv')
    audit.to_csv(args.destination / 'receipt_audit.csv', index=False)
    draw_board(board, guide, args.destination / 'board.png')
    counts = audit.groupby('Product', sort=False).Fair.agg(['sum', 'count'])
    counts.columns = ['fair_receipts', 'total_receipts']
    counts.to_csv(args.destination / 'receipt_counts.csv')
    metrics = {
        'receipts': len(audit),
        'fair_receipts': int(audit.Fair.sum()),
        'rejected_receipts': int((~audit.Fair).sum()),
        'checked_cells': int(board.to_numpy().sum()),
        'board_shape': list(board.shape),
    }
    (args.destination / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')

