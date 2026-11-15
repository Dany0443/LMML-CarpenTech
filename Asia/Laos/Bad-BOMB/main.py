from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.linalg import eigh
from scipy.ndimage import binary_erosion, binary_fill_holes, gaussian_filter, uniform_filter
from align import align_shell

ROOT = Path(__file__).resolve().parent
paths = sorted((ROOT / 'images').glob('*.png'))
if len(paths) < 28:
    raise ValueError('At least 28 images are required')
with ThreadPoolExecutor(max_workers=6) as pool:
    crops = np.array(list(pool.map(align_shell, paths)))
interior = (crops[:, :, :, 1] > crops[:, :, :, 0] * 1.12).mean(axis=0) > 0.7
interior = binary_erosion(binary_fill_holes(interior), iterations=2)
features = crops[:, interior, :].reshape(len(paths), -1)
features = (features - features.mean(axis=1, keepdims=True)) / features.std(axis=1, keepdims=True)
mean = features.mean(axis=0)
centered = features - mean
eigenvalues, basis = eigh(centered @ centered.T, subset_by_index=(len(paths) - 3, len(paths) - 1))
components = basis.T @ centered / np.sqrt(eigenvalues[:, None])
residual = centered - basis @ (basis.T @ centered)
heatmaps = np.zeros(crops.shape[:3], dtype=np.float32)
heatmaps[:, interior] = (residual.reshape(len(paths), -1, 3) ** 2).mean(axis=2)
heatmaps = gaussian_filter(heatmaps, sigma=(0, 1, 1))
flattened = heatmaps.reshape(len(paths), -1)
scores = np.partition(flattened, -30, axis=1)[:, -30:].mean(axis=1)
red_ratio = crops[:, :, :, 0] / (crops[:, :, :, 1] + 0.02)
red_scores = uniform_filter(red_ratio, size=(1, 3, 3))[:, 40:131, 12:44].max(axis=(1, 2))
selected = set(np.flatnonzero(red_scores > 1.01).tolist())
if len(selected) > 28:
    selected = set(np.argsort(-red_scores)[:28].tolist())
chromatic_count = len(selected)
for index in np.argsort(-scores):
    if len(selected) == 28:
        break
    selected.add(int(index))
lines = ['images/' + paths[index].name for index in sorted(selected)]
assert len(lines) == len(set(lines)) == 28
(ROOT / 'test-output.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
reports = ROOT / 'results'
reports.mkdir(exist_ok=True)
np.savez_compressed(reports / 'pca-model.npz', mean=mean, components=components, interior=interior)
with (reports / 'scores.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['filename', 'pca_score', 'red_score', 'selected'])
    writer.writerows(((path.name, float(scores[index]), float(red_scores[index]), index in selected) for index, path in enumerate(paths)))
summary = {'images': len(paths), 'selected': len(lines), 'chromatic_candidates': chromatic_count, 'pca_additions': 28 - chromatic_count, 'pca_components': 3, 'labels_used': False, 'ground_truth_accuracy': None}
(reports / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
canvas = Image.new('RGB', (7 * 154, 4 * 302), 'white')
draw = ImageDraw.Draw(canvas)
for position, index in enumerate(sorted(selected)):
    panel = Image.fromarray((crops[index] * 255).astype(np.uint8)).resize((98, 280))
    left, top = (position % 7 * 154, position // 7 * 302)
    canvas.paste(panel, (left + 28, top))
    draw.text((left + 8, top + 283), paths[index].name, fill='black')
canvas.save(reports / 'selected-shells.jpg', quality=95)
print(json.dumps(summary, indent=2))
