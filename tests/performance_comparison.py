#!/usr/bin/env python3
"""融合 kernel 性能测试。在包根目录执行：PYTHONPATH=core python tests/performance_comparison.py"""

import numpy as np
import ctypes
import time
import os
import sys

# 包根目录
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

try:
    import torch
    HAS_TORCH = torch.cuda.is_available()
except Exception:
    HAS_TORCH = False


class KernelWrapper:
    def __init__(self, lib_path):
        if not os.path.exists(lib_path):
            raise FileNotFoundError(lib_path)
        self.lib = ctypes.CDLL(lib_path)
        self.lib.fused_forward.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        self.lib.fused_forward.restype = None

    def forward(self, audio, text):
        bs, seq_len, feat_dim = audio.shape
        audio = audio.astype(np.float32)
        text = text.astype(np.float32)
        sdp = np.zeros((bs, feat_dim), dtype=np.float32)
        dp = np.zeros((bs, seq_len), dtype=np.float32)
        self.lib.fused_forward(
            audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            text.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            sdp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            dp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            bs, seq_len, feat_dim, feat_dim,
        )
        return sdp, dp


def benchmark(wrapper, audio, text, warmup=20, iterations=200):
    for _ in range(warmup):
        wrapper.forward(audio, text)
    if HAS_TORCH:
        torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        if HAS_TORCH:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        wrapper.forward(audio, text)
        if HAS_TORCH:
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "p95": float(np.percentile(times, 95)),
        "throughput": len(audio) * 1000 / np.mean(times),
    }


def main():
    fused_so = os.path.join(ROOT, "kernels", "fused_sdp_dp_optimized.so")
    if not os.path.exists(fused_so):
        print(f"ERROR: Not found: {fused_so}")
        print("Run: cd kernels && bash build.sh")
        return
    print("Performance: Fused Kernel")
    print("=" * 60)
    wrapper = KernelWrapper(fused_so)
    configs = [
        {"batch_size": 1, "seq_len": 128, "feat_dim": 256},
        {"batch_size": 4, "seq_len": 128, "feat_dim": 256},
        {"batch_size": 8, "seq_len": 128, "feat_dim": 256},
        {"batch_size": 32, "seq_len": 128, "feat_dim": 256},
    ]
    for cfg in configs:
        bs, seq_len, feat_dim = cfg["batch_size"], cfg["seq_len"], cfg["feat_dim"]
        np.random.seed(42)
        audio = np.random.randn(bs, seq_len, feat_dim).astype(np.float32)
        text = np.random.randn(bs, seq_len, feat_dim).astype(np.float32)
        stats = benchmark(wrapper, audio, text)
        print(f"  batch={bs}: {stats['mean']:.4f}ms ± {stats['std']:.4f}ms  {stats['throughput']:.1f} samples/s")
    print("=" * 60)
    print("OK")


if __name__ == "__main__":
    main()
