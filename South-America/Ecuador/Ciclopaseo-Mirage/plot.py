import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
report = json.loads((ROOT / 'results/training-check.json').read_text())
with Image.open(ROOT / 'data/train-input.png') as image:
    source = image.convert('RGB')
with Image.open(ROOT / 'results/train-adversarial.png') as image:
    adversarial = image.convert('RGB')
difference = np.array(adversarial, dtype=np.int16) - np.array(source, dtype=np.int16)
amplified = Image.fromarray(np.clip(128 + difference * 100, 0, 255).astype(np.uint8))
canvas = Image.new('RGB', (752, 340), 'white')
draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default(size=16)
small = ImageFont.load_default(size=13)
panels = [(source, 'Original: mountain bike'), (adversarial, 'Modified: moped'),
          (amplified, 'Pixel changes, amplified 100x')]
for index, (panel, title) in enumerate(panels):
    left = 20 + 244 * index
    canvas.paste(panel, (left, 43))
    draw.text((left, 16), title, font=small, fill='black')
draw.text((20, 279), f"Moped confidence: {report['source_prediction']['target_confidence']:.2%} to "
          f"{report['saved_prediction']['target_confidence']:.4%}; maximum channel change: 1 / 255", font=font, fill='black')
draw.text((20, 309), 'Only the third panel exaggerates changes. The saved adversarial image is shown unchanged.',
          font=small, fill='#444444')
canvas.save(ROOT / 'results/comparison.png')
