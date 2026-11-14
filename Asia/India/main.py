import csv
import os
import struct
import sys
import tarfile
import zipfile
import zlib
from collections import defaultdict

import cv2
import numpy as np

SIDE = 320
CAND_THR = 18
PER_IMAGE = 8
HD_SURE = 3
MIN_INL = 15
MIN_RATIO = 0.55
MIN_NCC = 0.85
MIN_COVER = 0.5
MAX_CLIQUE = 5
EXTS = (".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".webp", ".tif", ".tiff")
DEBUG = os.environ.get("DEBUG") == "1"

POP = np.array([bin(i).count("1") for i in range(65536)], dtype=np.uint8)
M16, S16, S32, S48 = np.uint64(0xFFFF), np.uint64(16), np.uint64(32), np.uint64(48)


def is_image(fn):
    return fn.lower().endswith(EXTS) and not fn.startswith(".")


def inflate_at(view, start):
    stream = zlib.decompressobj(-15)
    parts, pos, step = [], start, 1 << 20
    while pos < len(view):
        chunk = view[pos:pos + step]
        parts.append(stream.decompress(chunk))
        pos += len(chunk)
        if stream.eof:
            pos -= len(stream.unused_data)
            break
    return b"".join(parts), pos


def salvage_zip(path):
    # damaged or truncated zip: read the local entries one by one, no central directory needed
    with open(path, "rb") as fh:
        blob = fh.read()
    view = memoryview(blob)
    pos = blob.find(b"PK\x03\x04")
    while pos != -1:
        nxt = pos + 4
        try:
            flags, method = struct.unpack_from("<HH", blob, pos + 6)
            csize, _, nlen, xlen = struct.unpack_from("<IIHH", blob, pos + 18)
            name = os.path.basename(blob[pos + 30:pos + 30 + nlen].decode("utf-8", "replace"))
            start = pos + 30 + nlen + xlen
            if not flags & 1 and is_image(name):
                if method == 0:
                    body, nxt = bytes(view[start:start + csize]), start + csize
                elif method == 8:
                    body, nxt = inflate_at(view, start)
                else:
                    body = None
                if body:
                    yield name, body
        except (struct.error, zlib.error):
            pass
        pos = blob.find(b"PK\x03\x04", nxt)


def describe_cwd():
    rows = []
    for entry in sorted(os.listdir("."))[:30]:
        if os.path.isdir(entry):
            rows.append("%s: dir with %d entries" % (entry, len(os.listdir(entry))))
        else:
            with open(entry, "rb") as fh:
                head = fh.read(8)
            rows.append("%s: %d bytes, starts %s" % (entry, os.path.getsize(entry), head.hex()))
    return "; ".join(rows)


def iter_images(src):
    if os.path.isdir(src):
        for root, dirs, files in os.walk(src):
            dirs[:] = sorted(d for d in dirs if not d.startswith((".", "__")))
            for fn in sorted(files):
                if is_image(fn):
                    with open(os.path.join(root, fn), "rb") as fh:
                        yield fn, fh.read()
    elif os.path.isfile(src):
        if zipfile.is_zipfile(src):
            try:
                with zipfile.ZipFile(src) as zf:
                    for info in zf.infolist():
                        fn = os.path.basename(info.filename)
                        if not info.is_dir() and is_image(fn):
                            yield fn, zf.read(info)
                return
            except (zipfile.BadZipFile, zlib.error, NotImplementedError):
                pass
        elif tarfile.is_tarfile(src):
            with tarfile.open(src) as tf:
                for member in tf:
                    fn = os.path.basename(member.name)
                    if member.isfile() and is_image(fn):
                        yield fn, tf.extractfile(member).read()
            return
        yield from salvage_zip(src)


def source_options():
    for arg in sys.argv[1:]:
        if not arg.startswith("-"):
            yield arg
    here = os.getcwd()
    for entry in sorted(os.listdir(here)):
        if not entry.startswith("."):
            yield os.path.join(here, entry)
    yield here


def load_gray(raw):
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    h, w = img.shape
    k = SIDE / max(h, w)
    if k < 1:
        img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
    return img


def one_hash(view):
    low = cv2.dct(view)[:8, :8].ravel()
    return np.packbits(low > np.median(low[1:])).view(np.uint64)[0]


def hash_variants(img):
    small = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    views = [
        small,
        cv2.rotate(small, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(small, cv2.ROTATE_180),
        cv2.rotate(small, cv2.ROTATE_90_COUNTERCLOCKWISE),
        cv2.flip(small, 1),
    ]
    return [one_hash(v) for v in views]


def popcount(x):
    total = POP[x & M16] + POP[(x >> S16) & M16] + POP[(x >> S32) & M16] + POP[x >> S48]
    return total.astype(np.int16)


def find_candidates(hashes):
    n = len(hashes)
    base = hashes[:, 0]
    best = {}
    for start in range(0, n, 256):
        rows = base[start:start + 256, None]
        dist = np.full((len(rows), n), 99, np.int16)
        which = np.zeros((len(rows), n), np.int8)
        for v in range(5):
            d = popcount(rows ^ hashes[None, :, v])
            better = d < dist
            dist = np.where(better, d, dist)
            which = np.where(better, v, which)
        for r in range(len(rows)):
            i = start + r
            dist[r, i] = 99
            for j in np.argsort(dist[r])[:PER_IMAGE]:
                if dist[r, j] > CAND_THR:
                    break
                key = (min(i, int(j)), max(i, int(j)))
                if key not in best or dist[r, j] < best[key][0]:
                    best[key] = (int(dist[r, j]), which[r, j] == 4)
    return best


sift = cv2.SIFT_create(nfeatures=500)
matcher = cv2.BFMatcher(cv2.NORM_L2)
feat_cache = {}


def feats(idx, flip):
    key = (idx, flip)
    if key not in feat_cache:
        img = cv2.flip(gray[idx], 1) if flip else gray[idx]
        kps, des = sift.detectAndCompute(img, None)
        if des is None or len(kps) < 8:
            feat_cache[key] = None
        else:
            feat_cache[key] = (np.float32([k.pt for k in kps]), des.astype(np.uint8))
    return feat_cache[key]


def pair_stats(i, j, flip):
    out = {"good": 0, "inl": 0, "ratio": 0.0, "ncc": 0.0, "scale": 0.0, "cover": 0.0}
    a, b = feats(i, False), feats(j, flip)
    if a is None or b is None:
        return out
    knn = matcher.knnMatch(b[1].astype(np.float32), a[1].astype(np.float32), k=2)
    good = [p[0] for p in knn if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
    out["good"] = len(good)
    if len(good) < 6:
        return out
    src = np.float32([b[0][m.queryIdx] for m in good])
    dst = np.float32([a[0][m.trainIdx] for m in good])
    mat, mask = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=2000
    )
    if mat is None:
        return out
    out["inl"] = int(mask.sum())
    out["ratio"] = out["inl"] / len(good)
    out["scale"] = float(np.hypot(mat[0, 0], mat[1, 0]))

    img_a = gray[i]
    img_b = cv2.flip(gray[j], 1) if flip else gray[j]
    ha, wa = img_a.shape
    hb, wb = img_b.shape
    warped = cv2.warpAffine(img_b, mat, (wa, ha), flags=cv2.INTER_LINEAR)
    ones = np.full(img_b.shape, 255, np.uint8)
    inside = cv2.warpAffine(ones, mat, (wa, ha), flags=cv2.INTER_NEAREST) > 0
    inside = cv2.erode(inside.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
    area = inside.sum()
    out["cover"] = float(min(1.0, area / max(1.0, min(ha * wa, out["scale"] ** 2 * hb * wb))))
    if area < 200:
        return out
    pa = cv2.GaussianBlur(img_a, (5, 5), 0)[inside].astype(np.float32)
    pb = cv2.GaussianBlur(warped, (5, 5), 0)[inside].astype(np.float32)
    pa -= pa.mean()
    pb -= pb.mean()
    out["ncc"] = float((pa * pb).sum() / (np.linalg.norm(pa) * np.linalg.norm(pb) + 1e-6))
    return out


def accepted(s, hd):
    if s["good"] < 6 and hd <= HD_SURE:
        return True
    return (
        s["inl"] >= MIN_INL
        and s["ratio"] >= MIN_RATIO
        and s["ncc"] >= MIN_NCC
        and s["cover"] >= MIN_COVER
    )


def load_source(src):
    ids, imgs, rows = [], [], []
    for name, raw in iter_images(src):
        img = load_gray(raw)
        if img is not None:
            ids.append(name)
            imgs.append(img)
            rows.append(hash_variants(img))
    return ids, imgs, rows


names, gray, hash_rows = [], [], []
for option in source_options():
    names, gray, hash_rows = load_source(option)
    if names:
        print("images from", option)
        break
if not names:
    with open("output.csv", "w", newline="") as fh:
        csv.writer(fh).writerow(["image_a", "image_b"])
    raise SystemExit("no images found, argv=%s cwd=%s files: %s" % (sys.argv, os.getcwd(), describe_cwd()))
print(len(names), "images loaded")

cands = find_candidates(np.array(hash_rows, dtype=np.uint64)) if len(names) > 1 else {}
print(len(cands), "candidate pairs")

edges = []
debug_rows = []
for step, ((i, j), (hd, flip)) in enumerate(sorted(cands.items())):
    stats = pair_stats(i, j, flip)
    ok = accepted(stats, hd)
    if ok:
        edges.append((i, j))
    if DEBUG:
        debug_rows.append([names[i], names[j], hd, int(flip), stats["good"], stats["inl"],
                           round(stats["ratio"], 3), round(stats["ncc"], 3),
                           round(stats["scale"], 3), round(stats["cover"], 3), int(ok)])
    if step % 500 == 499:
        print(step + 1, "pairs checked")

parent = list(range(len(names)))


def root(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


for i, j in edges:
    parent[root(i)] = root(j)

groups = defaultdict(list)
for i in range(len(names)):
    groups[root(i)].append(i)

pairs = set()
for members in groups.values():
    if 1 < len(members) <= MAX_CLIQUE:
        pairs.update((min(a, b), max(a, b)) for a in members for b in members if a < b)
pairs.update((min(i, j), max(i, j)) for i, j in edges if len(groups[root(i)]) > MAX_CLIQUE)

with open("output.csv", "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["image_a", "image_b"])
    for a, b in sorted((tuple(sorted((names[i], names[j]))) for i, j in pairs)):
        writer.writerow([a, b])
print(len(pairs), "pairs written to output.csv")

if DEBUG:
    with open("candidates.csv", "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["a", "b", "hd", "flip", "good", "inl", "ratio", "ncc", "scale", "cover", "ok"])
        writer.writerows(debug_rows)
