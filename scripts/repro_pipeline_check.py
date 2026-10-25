"""Can this engine learn on real PlantVillage at all?

The Kaggle gate reached 0.08 validation accuracy after one epoch with the
restored (proven) recipe. That rules out the hyperparameters, so the fault is
structural. This runs the same engine, same augmentation, same normalisation, on
a small slice of the real dataset, on CPU, and reports whether the training loss
moves and whether validation accuracy beats chance.

Four arms, so the culprit is isolated rather than guessed:
  base          plain: rotation on, blocked shuffle
  no_aug        rotation and photometric jitter off
  random_shuffle  blocked -> random
  no_aug_random   both of the above

Chance for 38 classes is 0.026.
"""
import json, subprocess, sys, tempfile, csv
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

REPO = Path(__file__).resolve().parent
DEV = REPO / "data" / "manifests" / "plantvillage_development.csv"
IMAGES = REPO / "data" / "raw" / "PlantVillage"
SIZE = 128
PER_CLASS_TRAIN = 10
PER_CLASS_VAL = 5

dev = pd.read_csv(DEV)
train = dev[dev.split == "train"]
val = dev[dev.split == "validation"]
classes = sorted(train.label.unique().tolist())

def take(frame, per_class):
    return frame.groupby("label", group_keys=False).head(per_class).reset_index(drop=True)

tr, va = take(train, PER_CLASS_TRAIN), take(val, PER_CLASS_VAL)
print(f"classes {len(classes)} | train {len(tr)} | val {len(va)} | chance {1/len(classes):.3f}")

tmp = Path(tempfile.mkdtemp(prefix="repro_"))
def build(frame, tag):
    x = np.zeros((len(frame), SIZE, SIZE, 3), dtype=np.uint8)
    for i, rel in enumerate(frame.relative_path):
        with Image.open(IMAGES / "train" / rel) as im:
            x[i] = np.asarray(im.convert("RGB").resize((SIZE, SIZE), Image.Resampling.BILINEAR))
    np.save(tmp / f"{tag}_x.npy", x)
    np.save(tmp / f"{tag}_y.npy", np.array([classes.index(l) for l in frame.label], dtype=np.int64))
build(tr, "train"); build(va, "val")
(tmp / "classes.json").write_text(json.dumps(classes), encoding="utf-8")

def run(tag, rotation, shuffle, epochs=3):
    protocol = {
        "image_size": SIZE, "epochs": epochs, "patience": 99, "seeds": [42],
        "eval_batch_size": 32, "workers": 0, "shuffle_mode": shuffle,
        "rotation_deg": rotation, "grad_clip": 5.0,
        "warmup_fraction": 0.0, "min_lr_factor": 1.0,
        "label_smoothing": 0.0, "class_weight_alpha": 0.0, "ema_decay": 0.0,
        "architectures": {"efficientnet_b0": {"pretrained": True, "local_batch_size": 16,
                                              "lr": 0.0001, "weight_decay": 0.0001}},
    }
    pf = tmp / f"p_{tag}.json"; pf.write_text(json.dumps(protocol), encoding="utf-8")
    out = tmp / f"r_{tag}"
    cmd = [sys.executable, str(REPO / "scripts" / "sweep_worker.py"),
           "--arch", "efficientnet_b0", "--mode", "single", "--seeds", "42",
           "--device", "cpu", "--protocol", str(pf),
           "--train-images", str(tmp / "train_x.npy"), "--train-labels", str(tmp / "train_y.npy"),
           "--val-images", str(tmp / "val_x.npy"), "--val-labels", str(tmp / "val_y.npy"),
           "--classes-json", str(tmp / "classes.json"), "--output-root", str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stdout[-600:]); print(r.stderr[-1200:]); raise SystemExit(f"{tag} failed")
    with open(out / "efficientnet_b0" / "seed42" / "training_history.csv") as fh:
        rows = list(csv.DictReader(fh))
    return [(float(x["epoch"]), float(x["train_loss"]), float(x["val_accuracy"]), float(x["lr"])) for x in rows]

for tag, rot, shuf in (("base", 12.0, "blocked"),
                       ("noaug", 0.0, "blocked"),
                       ("random", 12.0, "random"),
                       ("noaug_random", 0.0, "random")):
    print(f"\n=== {tag}: rotation={rot} shuffle={shuf} ===")
    for ep, loss, acc, lr in run(tag, rot, shuf):
        print(f"  epoch {ep:.0f}  train_loss={loss:.4f}  val_acc={acc:.4f}  lr={lr:.2e}")
print(f"\n(scratch: {tmp})")
