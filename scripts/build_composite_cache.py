#!/usr/bin/env python3
"""Composite PlantVillage leaves onto field backgrounds, for S19.

The treatment arm for H3. Every training image keeps its leaf and its label, and
loses its laboratory backdrop. The model then has no consistent laboratory context
to lean on, which is the whole point of the experiment.

Design decisions, stated because they bound what the result can mean:

**One random background per image, fixed for the run.** Re-sampling every epoch
would be better augmentation but needs the engine to hold a background pool and a
per-image mask during training, which is a larger change to a working pipeline.
With a fixed random assignment the mapping from background to class is already
destroyed, because each class receives random backdrops. What is lost is
background diversity per image. That limitation goes in the report.

**Validation is not composited.** The PlantVillage validation score has to stay
comparable to the baseline, so only the training cache is altered. Field
evaluation on PlantDoc is untouched either way.

**The background pool is sampled once and recorded.** `background_index.npy`
stores which background each image received, so the treatment is reproducible.

    python scripts/build_composite_cache.py \
        --backgrounds data/raw/DeepWeeds \
        --masks data/derived/masks_train.npz \
        --train-cache /kaggle/temp/plantvillage_cache/train_224_...\
        --out /kaggle/temp/plantvillage_cache/train_composite_224_x.npy
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SIZE = 224
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_pool(directory: Path, limit: int) -> np.ndarray:
    """Read up to `limit` background images, resized to the training size."""
    files = [p for p in sorted(directory.rglob("*"))
             if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise SystemExit(f"no images under {directory}")
    if limit:
        files = files[:limit]
    pool = np.zeros((len(files), SIZE, SIZE, 3), dtype=np.uint8)
    unreadable = 0
    for i, path in enumerate(files):
        try:
            with Image.open(path) as image:
                pool[i] = np.asarray(image.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR))
        except Exception:
            unreadable += 1
    if unreadable:
        print(f"  {unreadable} background(s) unreadable, left black")
    print(f"background pool: {len(files)} images from {directory}")
    return pool


def unpack_masks(path: Path) -> np.ndarray:
    blob = np.load(path, allow_pickle=False)
    packed, n, size = blob["packed"], int(blob["n"]), int(blob["size"])
    if size != SIZE:
        raise SystemExit(f"masks were built at {size}px, expected {SIZE}")
    flat = np.unpackbits(packed, axis=1)[:, : size * size]
    return flat.reshape(n, size, size).astype(bool)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backgrounds", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--train-cache", type=Path, required=True,
                        help="the uint8 memmap the engine already trains on")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pool-limit", type=int, default=4000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0, help="stop early, for a smoke test")
    parser.add_argument("--min-fg", type=float, default=0.25,
                        help="below this foreground fraction the mask is unreliable")
    parser.add_argument("--max-fg", type=float, default=0.90,
                        help="above this the mask is unreliable, usually a full-frame texture")
    parser.add_argument("--check-sheet", type=Path, default=None)
    args = parser.parse_args()

    pool = load_pool(args.backgrounds, args.pool_limit)
    masks = unpack_masks(args.masks)
    source = np.load(args.train_cache, mmap_mode="r", allow_pickle=False)

    n = min(len(source), len(masks), args.limit or len(source))
    if len(masks) != len(source):
        raise SystemExit(
            f"mask count {len(masks)} does not match cache rows {len(source)}. "
            "Masks must be built from the same split in the same order."
        )
    print(f"compositing {n:,} training images over a pool of {len(pool)} backgrounds")

    out = np.lib.format.open_memmap(args.out, mode="w+", dtype=np.uint8,
                                    shape=(n, SIZE, SIZE, 3))
    rng = np.random.default_rng(args.seed)
    chosen = rng.integers(0, len(pool), size=n)
    started = time.perf_counter()

    # The corner-colour plus greenness heuristic fails on close-up leaf textures,
    # where the frame is almost entirely leaf. There the mask can claim 0.08 or
    # 1.00 foreground, and compositing would replace leaf tissue with field
    # background. Those images pass through unaltered rather than being trained
    # on in a corrupted form.
    coverage = masks[:n].mean(axis=(1, 2))
    trusted = (coverage >= args.min_fg) & (coverage <= args.max_fg)
    print(f"  trusted masks: {trusted.sum():,}/{n:,} ({trusted.mean():.1%}); "
          f"{n - trusted.sum():,} pass through unaltered")

    for i in range(n):
        image = np.asarray(source[i])
        if not trusted[i]:
            out[i] = image
            continue
        back = pool[chosen[i]]
        keep = masks[i][..., None]
        out[i] = np.where(keep, image, back)
        if (i + 1) % 5000 == 0:
            rate = (i + 1) / max(time.perf_counter() - started, 1e-6)
            print(f"  {i + 1:,}/{n:,}  {rate:.0f} img/s", flush=True)
    out.flush()
    del out

    np.save(Path(str(args.out)).with_suffix(".background_index.npy"), chosen)
    summary = {
        "images": n,
        "backgrounds_in_pool": len(pool),
        "backgrounds_directory": str(args.backgrounds),
        "masks": str(args.masks),
        "source_cache": str(args.train_cache),
        "output": str(args.out),
        "seed": args.seed,
        "policy": "one random background per image, fixed for the run; validation not composited",
        "mean_foreground_fraction": round(float(masks[:n].mean()), 4),
        "images_composited": int(trusted.sum()),
        "images_passed_through": int(n - trusted.sum()),
        "mask_trust_band": [args.min_fg, args.max_fg],
        "pass_through_reason": ("HSV and corner-colour masks are unreliable outside the trust "
                                "band, typically on close-up leaf textures; those images keep "
                                "their original background"),
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }
    Path(str(args.out) + ".json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    if args.check_sheet:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pool_idx = np.flatnonzero(trusted)
        skip_idx = np.flatnonzero(~trusted)
        picks = list(pool_idx[:4]) + list(skip_idx[:2])
        if len(picks) < 6:                      # pad so no panel is left empty
            picks += [i for i in pool_idx[4:] if i not in picks][: 6 - len(picks)]
        if not picks:
            picks = list(rng.choice(n, size=min(6, n), replace=False))
        fig, axes = plt.subplots(2, 6, figsize=(15, 5.4))
        for col, index in enumerate(picks):
            axes[0][col].imshow(np.asarray(source[index]))
            axes[0][col].set_title("before", fontsize=7)
            axes[1][col].imshow(np.asarray(np.load(args.out, mmap_mode="r", allow_pickle=False)[index]))
            axes[1][col].set_title("after", fontsize=7)
            for row in range(2):
                axes[row][col].axis("off")
        fig.suptitle("Composite check: leaf kept, laboratory backdrop replaced", fontsize=10)
        fig.tight_layout()
        Path(args.check_sheet).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.check_sheet, dpi=140)
        plt.close(fig)
        print(f"wrote {args.check_sheet}")


if __name__ == "__main__":
    main()
