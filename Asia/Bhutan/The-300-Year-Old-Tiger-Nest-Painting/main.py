import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.distance import cdist

paths = sorted(Path('input').glob('*.png'))
if not paths:
    raise ValueError('No PNG strips found in input/')
strips = [cv2.imread(str(path), cv2.IMREAD_COLOR) for path in paths]
if any(strip is None for strip in strips):
    raise ValueError('A strip could not be decoded')
if len({strip.shape for strip in strips}) != 1:
    raise ValueError('Strips must have the same dimensions')
count = len(strips)
right = np.array([strip[:, -1, :] for strip in strips], dtype=np.float32).reshape(count, -1)
left = np.array([strip[:, 0, :] for strip in strips], dtype=np.float32).reshape(count, -1)
cost = cdist(right, left, metric='sqeuclidean')
np.fill_diagonal(cost, np.inf)
successor = {}
predecessor = {}
parents = list(range(count))


def find(index):
    while parents[index] != index:
        parents[index] = parents[parents[index]]
        index = parents[index]
    return index


for edge in np.argsort(cost, axis=None):
    first, second = divmod(int(edge), count)
    if first in successor or second in predecessor or find(first) == find(second):
        continue
    successor[first] = second
    predecessor[second] = first
    parents[find(first)] = find(second)
    if len(successor) == count - 1:
        break
start = next(index for index in range(count) if index not in predecessor)
order = [start]
while order[-1] in successor:
    order.append(successor[order[-1]])
assert len(order) == len(set(order)) == count
reconstructed = np.concatenate([strips[index] for index in order], axis=1)
if not cv2.imwrite('reconstructed_image.png', reconstructed):
    raise OSError('Could not write reconstructed_image.png')
Path('reconstructed_order.json').write_text(json.dumps([paths[index].name for index in order], indent=2) + '\n')
