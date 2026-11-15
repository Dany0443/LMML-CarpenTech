import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from PIL import Image, ImageEnhance, ImageOps

ROOT = Path(__file__).resolve().parent
with Image.open(ROOT / 'data/train-input.png') as image:
    source = image.convert('RGB')
cases = {
    'resize_256': source.resize((256, 256), Image.Resampling.BICUBIC),
    'resize_448': source.resize((448, 448), Image.Resampling.BICUBIC),
    'flipped': ImageOps.mirror(source),
    'darkened': ImageEnhance.Brightness(source).enhance(0.7),
}
checks = []
for name, image in cases.items():
    with tempfile.TemporaryDirectory() as scratch:
        folder = Path(scratch)
        image.save(folder / f'{name}.png')
        report_path = ROOT / 'results/variants' / f'{name}.json'
        completed = subprocess.run([
            sys.executable, str(ROOT / 'verify.py'), '--input', str(folder / f'{name}.png'),
            '--report', str(report_path), '--image', str(folder / 'adversarial.png'),
        ], text=True, capture_output=True, check=True)
        report = json.loads(report_path.read_text())
        checks.append({'case': name, 'confidence': report['saved_prediction']['target_confidence'],
                       'max_pixel_delta': report['max_pixel_delta'], 'bytes': report['output_bytes'],
                       'wall_seconds': report['wall_seconds'], 'pass': report['pass']})
        print(name, report['pass'], flush=True)
with (ROOT / 'results/variants.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
    writer.writeheader()
    writer.writerows(checks)
