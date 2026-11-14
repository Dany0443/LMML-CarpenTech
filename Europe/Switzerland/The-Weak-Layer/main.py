from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier


pits = pd.read_csv('test-input.csv')
released = pits.pop('slab_released')
classifier = DecisionTreeClassifier(random_state=42).fit(pits, released)
tree = classifier.tree_
letters = []
pending = [0]

while pending:
    node = pending.pop()
    feature = tree.feature[node]
    if feature < 0:
        continue
    letters.append(pits.columns[feature][0].upper())
    pending.append(tree.children_right[node])
    pending.append(tree.children_left[node])

Path('test-output.txt').write_text('SIGMOID_' + ''.join(letters) + '\n', encoding='utf-8')
