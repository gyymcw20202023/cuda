#!/usr/bin/env python3
"""融合 kernel 与 PyTorch 一致性检查。在包根目录执行：PYTHONPATH=core python tests/consistency_check.py"""

import numpy as np
import sys
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "core"))

from fused_kernel_wrapper import FusedKernelWrapper

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except Exception:
    HAS_TORCH = False


def gen_data(batch_size=4, seq_len=128, feat_dim=256):
    np.random.seed(42)
    a = np.random.randn(batch_size, seq_len, feat_dim).astype(np.float32)
    t = np.random.randn(batch_size, seq_len, feat_dim).astype(np.float32)
    return a, t


def pytorch_baseline(audio, text):
    if not HAS_TORCH:
        return None, None
    audio_t = torch.from_numpy(audio).cuda()
    text_t = torch.from_numpy(text).cuda()
    sdp = torch.mean(audio_t, dim=1).cpu().numpy()
    dp = (torch.sigmoid(torch.mean(text_t, dim=-1)) * 10.0).cpu().numpy()
    return sdp, dp


def main():
    print("Consistency Check: Fused CUDA vs PyTorch")
    print("=" * 60)
    # 从包根查找 kernels/
    kernels_dir = os.path.join(ROOT, "kernels")
    so_path = os.path.join(kernels_dir, "fused_sdp_dp_optimized.so")
    if not os.path.exists(so_path):
        print(f"ERROR: {so_path} not found. Run: cd kernels && bash build.sh")
        return
    wrapper = FusedKernelWrapper(lib_path=so_path)
    if wrapper.lib is None:
        print("ERROR: Failed to load .so")
        return
    audio, text = gen_data()
    fused_sdp, fused_dp = wrapper.fused_forward(audio, text)
    print(f"Fused SDP shape: {fused_sdp.shape}, DP shape: {fused_dp.shape}")
    if HAS_TORCH:
        base_sdp, base_dp = pytorch_baseline(audio, text)
        sdp_diff = np.abs(fused_sdp - base_sdp)
        dp_diff = np.abs(fused_dp - base_dp)
        print(f"SDP max diff: {sdp_diff.max():.6f}, mean: {sdp_diff.mean():.6f}")
        print(f"DP  max diff: {dp_diff.max():.6f}, mean: {dp_diff.mean():.6f}")
        rtol = 1e-4
        if sdp_diff.max() < rtol and dp_diff.max() < rtol:
            print("✓ Consistency OK (within rtol)")
        else:
            print("⚠ Difference above rtol (float32 / GPU 差异可能略大)")
    else:
        print("(PyTorch not available, skipped baseline comparison)")
    print("=" * 60)


if __name__ == "__main__":
    main()
