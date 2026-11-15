from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from challenge_model import MAX_OUTPUT_BYTES, MAX_PIXEL_DELTA, MIN_TARGET_CONFIDENCE, TARGET_CLASS_ID, load_model, normalize_batch, pil_to_tensor, predict_tensor
MODEL_PATH = Path('classifier/mobilenet_v3_large-5c1a4163.pth')
GOOD_ENOUGH = 0.999
TIME_BUDGET_S = 40.0
TARGET = torch.tensor([TARGET_CLASS_ID])

def integer_image(src255: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
    d_int = delta + (delta.round() - delta).detach()
    return (src255 + d_int).clamp(0.0, 255.0)

def to_uint8(src255: torch.Tensor, delta_int: torch.Tensor) -> np.ndarray:
    out = (src255 + delta_int).clamp(0, 255).round().to(torch.uint8)
    return out[0].permute(1, 2, 0).numpy()

def pgd(model, src255, deadline, *, delta0=None, mask=None, stop_p=GOOD_ENOUGH, sched=(1.0, 0.5, 0.25), every=40):
    eps = float(MAX_PIXEL_DELTA)
    delta = torch.zeros_like(src255) if delta0 is None else delta0.clone()
    if mask is not None:
        delta = delta * mask
    momentum = torch.zeros_like(src255)
    best_p, best_delta, it = (-1.0, None, 0)
    while time.time() < deadline:
        step = sched[min(it // every, len(sched) - 1)]
        d = delta.clone().requires_grad_(True)
        logits = model(normalize_batch(integer_image(src255, d) / 255.0))
        F.cross_entropy(logits, TARGET).backward()
        with torch.no_grad():
            p_t = logits.softmax(1)[0, TARGET_CLASS_ID].item()
            if int(logits.argmax()) == TARGET_CLASS_ID and p_t > best_p:
                best_p, best_delta = (p_t, delta.round().clone())
            if best_p >= stop_p:
                break
            g = d.grad
            g = g / (g.abs().mean() + 1e-12)
            momentum = 0.9 * momentum + g
            delta = (delta - step * momentum.sign()).clamp(-eps, eps)
            if mask is not None:
                delta = delta * mask
        it += 1
    return (best_p, best_delta, it)

def solve(model, src_img: Image.Image, t0: float) -> np.ndarray:
    src255 = (pil_to_tensor(src_img) * 255.0).round()
    best_p, delta, iters = pgd(model, src255, t0 + TIME_BUDGET_S)
    print(f'iters={iters} p={best_p:.5f}', file=sys.stderr)
    if delta is None:
        delta = torch.zeros_like(src255)
    return to_uint8(src255, delta)

def main() -> int:
    t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='test-input.png')
    ap.add_argument('--output', default='test-output.png')
    args = ap.parse_args()
    torch.set_num_threads(2)
    model = load_model(MODEL_PATH)
    for prm in model.parameters():
        prm.requires_grad_(False)
    with Image.open(args.input) as im:
        src_img = im.convert('RGB')
    src_px = np.array(src_img, dtype=np.int16)
    out_px = solve(model, src_img, t0)
    Image.fromarray(out_px).save(args.output, format='PNG', optimize=True)
    with Image.open(args.output) as saved:
        assert saved.mode == 'RGB' and saved.size == src_img.size
        saved_px = np.array(saved, dtype=np.int16)
        pred = predict_tensor(model, pil_to_tensor(saved))
    max_delta = int(np.abs(saved_px - src_px).max())
    size = Path(args.output).stat().st_size
    ok = pred.class_id == TARGET_CLASS_ID and pred.target_confidence >= MIN_TARGET_CONFIDENCE and (max_delta <= MAX_PIXEL_DELTA) and (size <= MAX_OUTPUT_BYTES)
    print(f'class={pred.class_id} ({pred.class_name}) p_target={pred.target_confidence:.6f} max_delta={max_delta} bytes={size} time={time.time() - t0:.1f}s ok={ok}')
    return 0
if __name__ == '__main__':
    sys.exit(main())
