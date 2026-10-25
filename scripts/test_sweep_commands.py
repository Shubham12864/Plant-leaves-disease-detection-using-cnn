#!/usr/bin/env python3
"""Check the command lines sweep_worker builds, without launching anything.

This exists because a real Kaggle run failed on it. ``build_command`` used to
put ``sys.executable`` at the head of the argument list *and* again in front of
``torch.distributed.run``, so torchrun received the interpreter as its
training-script positional and tried to execute the Python binary as source:

    File "/usr/bin/python3", line 1
        ELF
    SyntaxError: source code cannot contain null bytes

Running the worker on CPU never caught it, because the CPU path takes the
``single`` branch and never touches torchrun. So this test asserts the shape of
both commands, which needs no GPU and no subprocess.

    python scripts/test_sweep_commands.py
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from sweep_worker import build_command  # noqa: E402

SCRIPT = Path("/kaggle/working/train_ddp.py")
PATHS = {
    "train_x": Path("/tmp/train_x.npy"), "train_y": Path("/tmp/train_y.npy"),
    "val_x": Path("/tmp/val_x.npy"), "val_y": Path("/tmp/val_y.npy"),
    "classes": Path("/tmp/classes.json"),
}
PROTOCOL = {
    "epochs": 12, "patience": 3, "eval_batch_size": 128, "workers": 4,
    "shuffle_mode": "random", "rotation_deg": 15.0, "grad_clip": 5.0,
    "warmup_fraction": 0.05, "label_smoothing": 0.05, "class_weight_alpha": 0.5,
    "ema_decay": 0.999, "image_size": 224,
}
CFG = {"local_batch_size": 64, "lr": 0.0003, "weight_decay": 0.05}

failures = []


def check(label, condition, detail=""):
    if condition:
        print(f"  PASS  {label}")
    else:
        print(f"  FAIL  {label} {detail}")
        failures.append(label)


def run(mode, device="cuda"):
    return build_command("efficientnet_b0", 42, CFG, PROTOCOL, PATHS,
                         Path("/kaggle/working/runs/efficientnet_b0/seed42"),
                         mode, SCRIPT, device, PROTOCOL["epochs"])


print("ddp mode")
ddp = run("ddp")
check("starts with the interpreter", ddp[0] == sys.executable, ddp[0])
check("then -m torch.distributed.run", ddp[1:3] == ["-m", "torch.distributed.run"], ddp[1:3])
check("interpreter appears exactly once",
      sum(1 for token in ddp if token == sys.executable) == 1,
      f"count={sum(1 for t in ddp if t == sys.executable)}")
check("script appears exactly once", ddp.count(str(SCRIPT)) == 1, f"count={ddp.count(str(SCRIPT))}")
check("uses --standalone", "--standalone" in ddp)
check("requests two processes", "--nproc_per_node=2" in ddp)
check("no --no-dist", "--no-dist" not in ddp)

# The failure that reached Kaggle: the token torchrun treats as the training
# entrypoint must be the script, never the interpreter.
index = ddp.index("--nproc_per_node=2")
entrypoint = ddp[index + 1]
check("torchrun entrypoint is the script", entrypoint == str(SCRIPT), entrypoint)
check("entrypoint is not the interpreter", entrypoint != sys.executable, entrypoint)
check("entrypoint is a python file", entrypoint.endswith(".py"), entrypoint)
check("no bare interpreter after the flags",
      not any(Path(t).is_absolute() and Path(t).name.startswith("python") and not t.endswith(".py")
              for t in ddp[3:]),
      [t for t in ddp[3:] if "python" in Path(t).name])

print("single mode")
single = run("single", device="cuda")
check("starts with interpreter then script", single[0] == sys.executable and single[1] == str(SCRIPT))
check("script appears exactly once", single.count(str(SCRIPT)) == 1)
check("interpreter appears exactly once",
      sum(1 for token in single if token == sys.executable) == 1)
check("no torchrun", "torch.distributed.run" not in single)
check("has --no-dist", "--no-dist" in single)
check("names the device", "--device" in single and "cuda" in single)
check("device is the last value", single[-1] == "cuda", single[-3:])

print("arguments survive in both modes")
for mode, command in (("ddp", ddp), ("single", single)):
    for flag, value in (("--arch", "efficientnet_b0"), ("--seed", "42"),
                        ("--epochs", "12"), ("--local-batch-size", "64"),
                        ("--class-weight-alpha", "0.5")):
        ok = flag in command and command[command.index(flag) + 1] == value
        check(f"{mode}: {flag} {value}", ok)

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    raise SystemExit(1)
print("COMMAND SHAPE OK")
