#!/usr/bin/env python3
"""Export Unified Gated Plant CNN to ONNX & Quantized INT8 with Edge Benchmarking.

Performs:
1. PyTorch to ONNX FP32 export with dynamic batching.
2. Dynamic INT8 quantization for edge / IoT / Android deployment.
3. Numerical verification against PyTorch FP32 baseline (tolerance check).
4. Rigorous CPU latency & throughput benchmarking (PyTorch vs ONNX vs INT8).
5. File size reduction analysis.
"""
from pathlib import Path
import json
import os
import time
import numpy as np
import pandas as pd

import torch
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from src.plant_disease.models.unified_gated_cnn import build_unified_plant_cnn

ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = ROOT / "reports" / "unified_extended" / "unified_gated_plant_cnn_extended_best.pt"
EXPORT_DIR = ROOT / "reports" / "unified_extended" / "export"
EXPORT_DIR.mkdir(parents=True, exist_ok=True)

ONNX_FP32_PATH = EXPORT_DIR / "unified_gated_plant_cnn.onnx"
ONNX_INT8_PATH = EXPORT_DIR / "unified_gated_plant_cnn_int8.onnx"

CROPS_14_COUNT = 14
DISEASES_38_COUNT = 38


def load_pytorch_model():
    model = build_unified_plant_cnn(num_crops=CROPS_14_COUNT, num_diseases=DISEASES_38_COUNT)
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


class ExportWrapper(torch.nn.Module):
    """Wrapper to guarantee strict tuple outputs for ONNX graph tracer."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor):
        d_logits, c_logits, mask = self.model(x, return_mask=True)
        return d_logits, c_logits, mask


def export_onnx_fp32(model, dummy_input: torch.Tensor):
    print(f"\n[1/4] Exporting PyTorch model to ONNX FP32 at {ONNX_FP32_PATH}...")
    wrapper = ExportWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper,
        dummy_input,
        str(ONNX_FP32_PATH),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["disease_logits", "crop_logits", "saliency_mask"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "disease_logits": {0: "batch_size"},
            "crop_logits": {0: "batch_size"},
            "saliency_mask": {0: "batch_size"},
        },
        dynamo=False
    )

    # Validate ONNX graph structure
    onnx_model = onnx.load(str(ONNX_FP32_PATH))
    onnx.checker.check_model(onnx_model)
    print("  ✅ ONNX FP32 model graph validated successfully!")


def quantize_onnx_int8():
    print(f"\n[2/4] Applying Dynamic INT8 Quantization -> {ONNX_INT8_PATH}...")
    quantize_dynamic(
        model_input=str(ONNX_FP32_PATH),
        model_output=str(ONNX_INT8_PATH),
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=True
    )
    print("  ✅ Dynamic INT8 quantization complete!")


def verify_numerical_parity(pt_model, dummy_input: torch.Tensor):
    print("\n[3/4] Verifying Numerical Parity between PyTorch and ONNX Runtime...")
    with torch.no_grad():
        pt_d, pt_c, pt_m = pt_model(dummy_input, return_mask=True)
        pt_d_np = pt_d.numpy()
        pt_c_np = pt_c.numpy()
        pt_m_np = pt_m.numpy()

    sess_fp32 = ort.InferenceSession(str(ONNX_FP32_PATH), providers=["CPUExecutionProvider"])
    ort_outs = sess_fp32.run(None, {"input": dummy_input.numpy()})
    onnx_d, onnx_c, onnx_m = ort_outs[0], ort_outs[1], ort_outs[2]

    diff_d = np.max(np.abs(pt_d_np - onnx_d))
    diff_c = np.max(np.abs(pt_c_np - onnx_c))
    diff_m = np.max(np.abs(pt_m_np - onnx_m))

    print(f"  Max Absolute Error (Disease Logits): {diff_d:.2e}")
    print(f"  Max Absolute Error (Crop Logits):    {diff_c:.2e}")
    print(f"  Max Absolute Error (Saliency Mask):  {diff_m:.2e}")

    assert diff_d < 1e-4, f"Disease logits discrepancy too high: {diff_d}"
    assert diff_c < 1e-4, f"Crop logits discrepancy too high: {diff_c}"
    print("  ✅ Numerical verification PASSED (Error < 1e-4)!")


def benchmark_latency(pt_model, dummy_input: torch.Tensor, iterations: int = 100):
    print(f"\n[4/4] Benchmarking CPU Inference Latency ({iterations} runs per runtime)...")
    np_input = dummy_input.numpy()

    sess_fp32 = ort.InferenceSession(str(ONNX_FP32_PATH), providers=["CPUExecutionProvider"])
    sess_int8 = ort.InferenceSession(str(ONNX_INT8_PATH), providers=["CPUExecutionProvider"])

    # Warmup
    for _ in range(10):
        with torch.no_grad():
            pt_model(dummy_input)
        sess_fp32.run(None, {"input": np_input})
        sess_int8.run(None, {"input": np_input})

    # Benchmark PyTorch FP32
    pt_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        with torch.no_grad():
            pt_model(dummy_input)
        pt_times.append((time.perf_counter() - t0) * 1000.0)

    # Benchmark ONNX FP32
    onnx_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        sess_fp32.run(None, {"input": np_input})
        onnx_times.append((time.perf_counter() - t0) * 1000.0)

    # Benchmark ONNX INT8
    int8_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        sess_int8.run(None, {"input": np_input})
        int8_times.append((time.perf_counter() - t0) * 1000.0)

    # File sizes
    pt_size_mb = os.path.getsize(CHECKPOINT_PATH) / (1024 * 1024)
    onnx_size_mb = os.path.getsize(ONNX_FP32_PATH) / (1024 * 1024)
    int8_size_mb = os.path.getsize(ONNX_INT8_PATH) / (1024 * 1024)

    results = [
        {
            "Runtime": "PyTorch FP32 (Base)",
            "Model Size (MB)": f"{pt_size_mb:.1f} MB",
            "Mean Latency (ms)": np.mean(pt_times),
            "P95 Latency (ms)": np.percentile(pt_times, 95),
            "Throughput (FPS)": 1000.0 / np.mean(pt_times),
            "Size Reduction": "0.0%"
        },
        {
            "Runtime": "ONNX Runtime FP32",
            "Model Size (MB)": f"{onnx_size_mb:.1f} MB",
            "Mean Latency (ms)": np.mean(onnx_times),
            "P95 Latency (ms)": np.percentile(onnx_times, 95),
            "Throughput (FPS)": 1000.0 / np.mean(onnx_times),
            "Size Reduction": f"{(1 - onnx_size_mb / pt_size_mb)*100:.1f}%"
        },
        {
            "Runtime": "ONNX Runtime INT8 (Edge Optimized)",
            "Model Size (MB)": f"{int8_size_mb:.1f} MB",
            "Mean Latency (ms)": np.mean(int8_times),
            "P95 Latency (ms)": np.percentile(int8_times, 95),
            "Throughput (FPS)": 1000.0 / np.mean(int8_times),
            "Size Reduction": f"{(1 - int8_size_mb / pt_size_mb)*100:.1f}%"
        }
    ]

    df = pd.DataFrame(results)
    df.to_csv(EXPORT_DIR / "edge_latency_benchmark.csv", index=False)
    print("\n" + df.to_string(index=False))

    summary = {
        "pytorch_size_mb": float(pt_size_mb),
        "onnx_fp32_size_mb": float(onnx_size_mb),
        "onnx_int8_size_mb": float(int8_size_mb),
        "onnx_int8_size_reduction_pct": float((1 - int8_size_mb / pt_size_mb) * 100),
        "pytorch_latency_ms": float(np.mean(pt_times)),
        "onnx_fp32_latency_ms": float(np.mean(onnx_times)),
        "onnx_int8_latency_ms": float(np.mean(int8_times)),
        "speedup_factor": float(np.mean(pt_times) / np.mean(int8_times))
    }
    with open(EXPORT_DIR / "edge_benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ All export artifacts and benchmark logs saved to {EXPORT_DIR}")


def main():
    print("==================================================")
    print("EDGE EXPORT & LATENCY OPTIMIZATION PIPELINE")
    print("==================================================")

    model = load_pytorch_model()
    dummy_input = torch.randn(1, 3, 224, 224, dtype=torch.float32)

    export_onnx_fp32(model, dummy_input)
    quantize_onnx_int8()
    verify_numerical_parity(model, dummy_input)
    benchmark_latency(model, dummy_input, iterations=50)


if __name__ == "__main__":
    main()
