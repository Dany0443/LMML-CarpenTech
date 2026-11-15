import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large

from align import align_shell

ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True, type=Path)
args = parser.parse_args()
torch.set_num_threads(2)
paths = sorted((ROOT / 'images').glob('*.png'))
with ThreadPoolExecutor(max_workers=6) as pool:
    crops = np.array(list(pool.map(align_shell, paths)))
model = mobilenet_v3_large(weights=None).eval()
model.load_state_dict(torch.load(args.checkpoint, map_location='cpu', weights_only=True))
mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
embeddings = []
with torch.inference_mode():
    for start in range(0, len(crops), 32):
        batch = torch.from_numpy(crops[start:start + 32]).permute(0, 3, 1, 2)
        batch = F.interpolate(batch, size=(320, 112), mode='bilinear', align_corners=False)
        batch = (batch - mean) / std
        for layer, block in enumerate(model.features):
            batch = block(batch)
            if layer == 3:
                early = F.adaptive_avg_pool2d(batch, (20, 7))
            if layer == 6:
                break
        embeddings.append(torch.cat([early, F.adaptive_avg_pool2d(batch, (20, 7))], dim=1).numpy())
features = np.concatenate(embeddings)
patch_scores = []
for row in range(features.shape[2]):
    for column in range(features.shape[3]):
        patch = torch.from_numpy(features[:, :, row, column])
        center = patch.median(dim=0).values
        spread = torch.clamp((patch - center).abs().median(dim=0).values, min=0.1)
        patch = (patch - center) / spread
        distances = torch.cdist(patch, patch) ** 2
        distances.fill_diagonal_(float('inf'))
        patch_scores.append(distances.topk(15, largest=False).values.mean(1).numpy())
patch_scores = np.array(patch_scores).T
scores = np.sort(patch_scores, axis=1)[:, -5:].mean(axis=1)
prior = set((ROOT / 'results/submission-1.txt').read_text().splitlines())
selected = {index for index, path in enumerate(paths) if 'images/' + path.name in prior}
assert len(selected) == 28
removed = sorted(selected, key=lambda index: scores[index])[:2]
added = [int(index) for index in np.argsort(-scores) if int(index) not in selected][:2]
selected.difference_update(removed)
selected.update(added)
text = ''.join('images/' + paths[index].name + '\n' for index in sorted(selected))
(ROOT / 'test-output.txt').write_text(text)
(ROOT / 'results/submission-2.txt').write_text(text)
with (ROOT / 'results/deep-scores.csv').open('w', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['filename', 'patch_distance', 'submission_1', 'submission_2'])
    writer.writerows((path.name, float(scores[index]), 'images/' + path.name in prior, index in selected)
                     for index, path in enumerate(paths))
changes = {'removed': [paths[index].name for index in removed], 'added': [paths[index].name for index in added],
           'method': 'Replace two weakest initial choices with the two strongest unselected images under the independent patch-feature ranking.'}
(ROOT / 'results/submission-2-changes.json').write_text(json.dumps(changes, indent=2) + '\n')
canvas = Image.new('RGB', (7 * 154, 4 * 302), 'white')
draw = ImageDraw.Draw(canvas)
for position, index in enumerate(sorted(selected)):
    panel = Image.fromarray((crops[index] * 255).astype(np.uint8)).resize((98, 280))
    left, top = position % 7 * 154, position // 7 * 302
    canvas.paste(panel, (left + 28, top))
    draw.text((left + 8, top + 283), paths[index].name, fill='black')
canvas.save(ROOT / 'results/selected-shells.jpg', quality=95)
print(json.dumps(changes, indent=2))
