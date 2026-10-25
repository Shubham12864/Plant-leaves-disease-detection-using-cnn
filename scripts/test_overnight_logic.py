#!/usr/bin/env python3
"""Prove the overnight notebook survives the things that waste a night.

The notebook's value is not that it trains models - it is that a broken model, a
crashed run, a hang, an exhausted budget or a dead session does not cost the
whole session. This executes the real notebook cells against stubbed worker
processes and checks what they actually do:

  1. gate failed        -> sweep does not start, results still ship
  2. one run fails      -> the loop continues, the failure is recorded
  3. one run hangs      -> TimeoutExpired is caught, the loop continues
  4. seed already done  -> skipped, not retrained
  5. budget exhausted   -> no new runs are started
  6. partial results    -> a zip is still produced

    python scripts/test_overnight_logic.py
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # plt.show() must not block a headless test
import matplotlib.pyplot as plt   # noqa: E402
import pandas as pd

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "08_Overnight_Sweep.ipynb"

cells = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["cells"]
SWEEP_SRC = next("".join(c["source"]) for c in cells
                 if c["cell_type"] == "code" and "GATE_PASSED" in "".join(c["source"])
                 and "PLAN" in "".join(c["source"]))
RESULTS_SRC = next("".join(c["source"]) for c in cells
                   if c["cell_type"] == "code" and "shutil.make_archive" in "".join(c["source"]))

failures = []


def check(label, condition, detail=""):
    print(f"  {'PASS' if condition else 'FAIL'}  {label}" + (f"  {detail}" if not condition else ""))
    if not condition:
        failures.append(label)


class FakeTimeout(Exception):
    pass


class FakeSubprocess:
    """Stands in for subprocess.run; behaviour is driven by `script`."""

    TimeoutExpired = FakeTimeout

    def __init__(self, script, output_root, epochs):
        self.script = dict(script)      # (arch, seed) -> "ok" | "fail" | "timeout"
        self.output_root = output_root
        self.epochs = epochs
        self.calls = []

    def run(self, command, env=None, check=False, timeout=None):
        arch = command[command.index("--arch") + 1]
        seed = int(command[command.index("--seeds") + 1])
        self.calls.append((arch, seed))
        behaviour = self.script.get((arch, seed), "ok")
        if behaviour == "timeout":
            raise FakeTimeout()
        if behaviour == "fail":
            return type("R", (), {"returncode": 1})()
        out = self.output_root / arch / f"seed{seed}"
        out.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([{"epoch": e, "train_loss": 0.5 / e, "val_accuracy": 0.9, "lr": 1e-4,
                       "val_macro_f1": 0.88}
                      for e in range(1, self.epochs + 1)]).to_csv(out / "training_history.csv", index=False)
        (out / "validation_metrics.json").write_text(json.dumps({
            "model": arch, "val_accuracy": 0.9, "val_macro_f1": 0.88,
            "best_epoch": self.epochs, "train_minutes": 20.0,
            "dataset_fingerprint": "a" * 64, "global_batch_size": 32, "weight_decay": 1e-4,
        }), encoding="utf-8")
        return type("R", (), {"returncode": 0})()


def run_sweep(tmp, script, gate_passed, budget_h=10.0, started_offset_h=0.0, pre_done=()):
    output_root = tmp / "runs"
    output_root.mkdir(parents=True, exist_ok=True)
    for arch, seed in pre_done:
        d = output_root / arch / f"seed{seed}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "validation_metrics.json").write_text('{"model": "%s", "val_macro_f1": 0.9}' % arch)
    fake = FakeSubprocess(script, output_root, epochs=2)
    env = {
        "GATE_PASSED": gate_passed,
        "PROTOCOL_PATH": tmp / "protocol.json",
        "OUTPUT_ROOT": output_root,
        "LOG_PATH": tmp / "_overnight_log.json",
        "STARTED_AT": time.time() - started_offset_h * 3600,
        "SESSION_BUDGET_H": budget_h,
        "RESERVE_H": 0.75,
        "ALLOW_EXTRA_SEEDS": False,
        "MINUTES_PER_EPOCH": 1.0,          # fake: 1 min/epoch
        "PILOT_MINUTES": {"smallcnn": 0.67, "resnet18": 0.78,
                          "efficientnet_b0": 2.45, "convnext_tiny": 3.15},
        "subprocess": fake,
        "time": time,
        "json": json,
        "os": __import__("os"),
        "sys": __import__("sys"),
        "Path": Path,
        "print": print,
        "worker_command": lambda arch, mode, seeds, epochs, out: [
            "python", "worker", "--arch", arch, "--mode", mode,
            "--seeds", *[str(s) for s in seeds], "--epochs", str(epochs),
            "--output-root", str(out),
        ],
    }

    def _log(entry):
        path = tmp / "_overnight_log.json"
        entries = json.loads(path.read_text()) if path.is_file() else []
        entries.append(entry)
        path.write_text(json.dumps(entries))

    env["log"] = _log
    (tmp / "protocol.json").write_text(json.dumps({
        "seeds": [42, 43], "epochs": 2,
        "architectures": {"efficientnet_b0": {}, "convnext_tiny": {}, "resnet18": {}, "smallcnn": {}},
    }))
    exec(compile(SWEEP_SRC, "sweep_cell", "exec"), env)
    return fake, env


def run_results(tmp):
    output_root = tmp / "runs"
    env = {
        "OUTPUT_ROOT": output_root,
        "STARTED_AT": time.time(),
        "shutil": shutil,
        "Path": Path,
        "pd": pd,
        "plt": plt,
        "json": json,
        "time": time,
        "print": print,
        "display": lambda x: None,
    }
    # the cell writes to a fixed /kaggle path; capture the name instead
    src = RESULTS_SRC.replace('zip_name = "/kaggle/working/overnight_runs"',
                              f'zip_name = r"{tmp / "overnight_runs"}"')
    exec(compile(src, "results_cell", "exec"), env)
    return tmp / "overnight_runs.zip"


print("1. gate failed -> sweep must not start, results must still ship")
tmp = Path(tempfile.mkdtemp(prefix="on1_"))
fake, _ = run_sweep(tmp, {}, gate_passed=False)
check("no worker invoked", fake.calls == [], fake.calls)
zip_path = run_results(tmp)
check("zip still produced", zip_path.is_file())

print("\n2. one run fails, one hangs -> loop continues")
tmp = Path(tempfile.mkdtemp(prefix="on2_"))
script = {("efficientnet_b0", 42): "ok", ("efficientnet_b0", 43): "fail",
          ("convnext_tiny", 42): "timeout", ("convnext_tiny", 43): "ok"}
fake, _ = run_sweep(tmp, script, gate_passed=True)
check("all planned runs attempted", len(fake.calls) == 8, fake.calls)
check("failure did not abort the loop", ("smallcnn", 43) in fake.calls, fake.calls)
log = json.loads((tmp / "_overnight_log.json").read_text())
statuses = {e.get("status") for e in log if e.get("stage") == "run"}
check("failed recorded", "failed" in statuses, statuses)
check("timeout recorded", "timeout" in statuses, statuses)
check("successes recorded", "ok" in statuses, statuses)
metrics = list((tmp / "runs").rglob("validation_metrics.json"))
check("6 of 8 runs produced metrics", len(metrics) == 6, len(metrics))

print("\n3. finished seeds are skipped, not retrained")
tmp = Path(tempfile.mkdtemp(prefix="on3_"))
fake, _ = run_sweep(tmp, {}, gate_passed=True, pre_done=[("efficientnet_b0", 42)])
check("skipped the completed seed", ("efficientnet_b0", 42) not in fake.calls, fake.calls)
check("ran the remaining seed", ("efficientnet_b0", 43) in fake.calls)

print("\n4. budget exhausted -> no new runs started")
tmp = Path(tempfile.mkdtemp(prefix="on4_"))
fake, _ = run_sweep(tmp, {}, gate_passed=True, budget_h=2.0, started_offset_h=1.9)
check("started nothing when out of budget", fake.calls == [], fake.calls)

print("\n5. partial results still produce a usable zip")
tmp = Path(tempfile.mkdtemp(prefix="on5_"))
run_sweep(tmp, {("efficientnet_b0", 42): "ok"}, gate_passed=True)
zip_path = run_results(tmp)
with zipfile.ZipFile(zip_path) as zf:
    names = zf.namelist()
check("zip contains the finished run", any("efficientnet_b0" in n for n in names), names[:5])
check("zip contains comparison.csv", "comparison.csv" in names, names[:8])

print()
if failures:
    print(f"{len(failures)} CHECK(S) FAILED: {failures}")
    raise SystemExit(1)
print("OVERNIGHT LOGIC OK — every failure path survives")
