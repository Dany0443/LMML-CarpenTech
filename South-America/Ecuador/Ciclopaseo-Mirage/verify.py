import argparse
import hashlib
import json
from pathlib import Path
import platform
import resource
import subprocess
import sys
import tempfile
import time

import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.models import mobilenet_v3_large


ROOT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, default=ROOT / 'data/train-input.png')
parser.add_argument('--report', type=Path, default=ROOT / 'results/training-check.json')
parser.add_argument('--image', type=Path, default=ROOT / 'results/train-adversarial.png')
args = parser.parse_args()
args.report.parent.mkdir(parents=True, exist_ok=True)
args.image.parent.mkdir(parents=True, exist_ok=True)
weights = ROOT / 'classifier/mobilenet_v3_large-5c1a4163.pth'
with tempfile.TemporaryDirectory() as scratch:
    folder = Path(scratch)
    (folder / 'classifier').symlink_to(ROOT / 'classifier', target_is_directory=True)
    (folder / 'test-input.png').write_bytes(args.input.read_bytes())
    before = time.monotonic()
    completed = subprocess.run([sys.executable, str(ROOT / 'main.py')], cwd=folder,
                               text=True, capture_output=True, timeout=60, check=True)
    elapsed = time.monotonic() - before
    args.image.write_bytes((folder / 'test-output.png').read_bytes())
    extra_files = sorted(path.name for path in folder.iterdir()
                         if path.name not in {'classifier', 'test-input.png', 'test-output.png'})

with Image.open(args.input) as image:
    source = np.array(image.convert('RGB'), dtype=np.int16)
    dimensions = image.size
with Image.open(args.image) as image:
    mode, encoding, saved_dimensions = image.mode, image.format, image.size
    pixels = np.array(image, dtype=np.int16)
assert pixels.shape == source.shape
maximum = int(np.abs(pixels - source).max())
model = mobilenet_v3_large(weights=None).eval()
model.load_state_dict(torch.load(weights, weights_only=True, map_location='cpu'), strict=True)
torch.set_num_threads(2)
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def classify(rgb):
    batch = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1).unsqueeze(0) / 255
    with torch.inference_mode():
        probabilities = model((batch - mean) / std).softmax(1)[0]
    return {'class_id': int(probabilities.argmax()), 'target_confidence': float(probabilities[665])}


prediction = classify(pixels)
checks = {
    'class': prediction['class_id'] == 665,
    'confidence': prediction['target_confidence'] >= 0.99,
    'pixel_budget': maximum <= 1,
    'format': mode == 'RGB' and encoding == 'PNG' and saved_dimensions == dimensions,
    'size': args.image.stat().st_size <= 1024 * 1024,
    'runtime': elapsed <= 60,
    'memory': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss <= 2 * 1024 * 1024,
    'no_extra_files': not extra_files,
}
report = {
    'input': args.input.name,
    'source_prediction': classify(source),
    'saved_prediction': prediction,
    'max_pixel_delta': maximum,
    'changed_channels': int(np.count_nonzero(pixels - source)),
    'output_bytes': args.image.stat().st_size,
    'dimensions': list(saved_dimensions),
    'mode': mode,
    'format': encoding,
    'wall_seconds': elapsed,
    'child_peak_rss_mib': resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024,
    'checks': checks,
    'pass': all(checks.values()),
    'checkpoint_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
    'script_sha256': hashlib.sha256((ROOT / 'main.py').read_bytes()).hexdigest(),
    'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
    'png_sha256': hashlib.sha256(args.image.read_bytes()).hexdigest(),
    'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                 'torchvision': torchvision.__version__, 'numpy': np.__version__},
    'stdout': completed.stdout.strip(),
    'stderr': completed.stderr.strip(),
}
args.report.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
if not report['pass']:
    raise SystemExit(1)
