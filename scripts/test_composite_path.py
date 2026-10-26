"""End-to-end smoke test for the S19 compositing path, on real PlantVillage images."""
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
PY = sys.executable
N = 24

tmp = Path(tempfile.mkdtemp(prefix="comp_"))
dev = list(csv.DictReader((REPO / "data/manifests/plantvillage_development.csv").open(encoding="utf-8")))
train = [r for r in dev if r["split"] == "train"][:N]
root = REPO / "data/raw/PlantVillage/train"

cache = np.zeros((N, 224, 224, 3), dtype=np.uint8)
for i, row in enumerate(train):
    with Image.open(root / row["relative_path"]) as im:
        cache[i] = np.asarray(im.convert("RGB").resize((224, 224), Image.Resampling.BILINEAR))
np.save(tmp / "train_x.npy", cache)

sub = {"split": "train", "images": N,
       "paths_manifest": str(tmp / "small.csv"),
       "image_root": str(root)}
with (tmp / "small.csv").open("w", newline="", encoding="utf-8") as fh:
    w = csv.writer(fh)
    w.writerow(["dataset", "domain", "relative_path", "label", "sha256", "width", "height",
                "group_id", "source_id", "plant_id", "split"])
    for r in train:
        w.writerow([r["dataset"], r["domain"], r["relative_path"], r["label"], r["sha256"],
                    r["width"], r["height"], "", "", "", "train"])

print("1. build masks for the subset")
r = subprocess.run([PY, str(REPO / "scripts/make_leaf_masks.py"), "--split", "train",
                    "--manifest", str(tmp / "small.csv"), "--image-root", str(root),
                    "--out", str(tmp / "masks.npz"), "--check-sheet", str(tmp / "mask_check.png")],
                   cwd=REPO, capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-800:], r.stderr[-800:]); raise SystemExit(1)
summary = json.loads(r.stdout[r.stdout.index("{"):r.stdout.rindex("}") + 1])
print(f"   masks {summary['images']}, coverage {summary['mask_coverage_mean']}")

print("2. synthetic background pool (10 noise images, stand-ins for field photos)")
bg = tmp / "backgrounds"; bg.mkdir()
rng = np.random.default_rng(0)
for i in range(10):
    Image.fromarray(rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)).save(bg / f"bg{i:02d}.png")

print("3. composite")
r = subprocess.run([PY, str(REPO / "scripts/build_composite_cache.py"), "--backgrounds", str(bg),
                    "--masks", str(tmp / "masks.npz"), "--train-cache", str(tmp / "train_x.npy"),
                    "--out", str(tmp / "train_composite.npy"), "--seed", "42",
                    "--check-sheet", str(tmp / "composite_check.png")],
                   cwd=REPO, capture_output=True, text=True)
if r.returncode != 0:
    print(r.stdout[-1200:], r.stderr[-1200:]); raise SystemExit(1)
out_summary = json.loads((tmp / "train_composite.npy.json").read_text())
print(f"   composited {out_summary['images_composited']}, "
      f"passed through {out_summary['images_passed_through']}")

print("4. verify the output")
composite = np.load(tmp / "train_composite.npy", mmap_mode="r")
index = np.load(tmp / "train_composite.background_index.npy")
assert composite.shape == cache.shape, composite.shape
assert len(index) == N, len(index)
assert composite.dtype == np.uint8

masks = np.load(tmp / "masks.npz", allow_pickle=False)
m = np.unpackbits(masks["packed"], axis=1)[:, :224 * 224].reshape(N, 224, 224).astype(bool)
cov = m.mean(axis=(1, 2))
trusted = (cov >= 0.25) & (cov <= 0.90)

for i in range(N):
    before, after = np.asarray(cache[i]), np.asarray(composite[i])
    if torn := (before == after).all():
        pass
    if not trusted[i]:
        assert (before == after).all(), f"image {i} should have passed through unchanged"
    else:
        changed = (before != after).any(axis=2)
        # the leaf should survive: pixels inside the mask must be unchanged
        inside_changed = (changed & m[i]).sum()
        assert inside_changed == 0, f"image {i}: {inside_changed} leaf pixels were altered"
        assert changed.sum() > 0, f"image {i}: nothing was composited"

print("   leaf pixels preserved, background replaced, pass-through respected")
print(f"   backgrounds used: {len(set(index.tolist()))} distinct")
print("\nCOMPOSITE PATH OK")
print(f"(scratch: {tmp})")
