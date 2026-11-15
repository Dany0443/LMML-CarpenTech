import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

import cv2
import numpy as np
from skimage.metrics import structural_similarity

ROOT = Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as scratch:
    folder = Path(scratch)
    shutil.copyfile(ROOT / 'main.py', folder / 'main.py')
    shutil.copytree(ROOT / 'train-input', folder / 'input')
    started = time.monotonic()
    subprocess.run([sys.executable, 'main.py'], cwd=folder, check=True, timeout=20)
    elapsed = time.monotonic() - started
    order = json.loads((folder / 'reconstructed_order.json').read_text())
    truth = json.loads((ROOT / 'train-output/reconstructed_order.json').read_text())
    direct = sum(first == second for first, second in zip(order, truth)) / len(truth)
    reverse = sum(first == second for first, second in zip(order, reversed(truth))) / len(truth)
    expected = cv2.imread(str(ROOT / 'train-output/reconstructed_image.png'))
    reconstructed = cv2.imread(str(folder / 'reconstructed_image.png'))
    similarity = float(structural_similarity(expected, reconstructed, channel_axis=2, data_range=255))
    assert len(order) == len(set(order)) == len(truth)
    assert set(order) == set(truth)
    assert max(direct, reverse) >= 0.9
    assert similarity >= 0.9
    assert elapsed < 20
    for name in ['reconstructed_image.png', 'reconstructed_order.json']:
        shutil.copyfile(folder / name, ROOT / name)
    report = {'strips': len(order), 'image_shape': list(reconstructed.shape),
              'direct_order_accuracy': direct, 'reverse_order_accuracy': reverse,
              'ssim': similarity, 'pixel_identical': bool(np.array_equal(expected, reconstructed)),
              'runtime_seconds': elapsed, 'standalone': True, 'local_pass': True,
              'versions': {'python': sys.version.split()[0], 'numpy': np.__version__, 'opencv': cv2.__version__}}
(ROOT / 'results/training-check.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
