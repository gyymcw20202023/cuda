#!/usr/bin/env python3
"""
演示推理脚本：用 USE_FUSED=0 / USE_FUSED=1 跑两遍，对比总耗时。
没有正式推理脚本时，用本脚本做「全链路对比」的替代：只跑 SDP+DP 段，模拟「整段耗时」。

用法（在主项目根目录，主项目下有 sdp_dp.so）:
  cd /data3/test-bert-modifiled/bert-vits2-pt-modified
  export PYTHONPATH=".:/data3/test-bert-modifiled/test1/bert-vits2-fused-module-delivery/core:$PYTHONPATH"
  USE_FUSED=0 python3 /data3/test-bert-modifiled/test1/bert-vits2-fused-module-delivery/demo_inference.py
  USE_FUSED=1 python3 /data3/test-bert-modifiled/test1/bert-vits2-fused-module-delivery/demo_inference.py
  对比两次输出的「总耗时(ms)」。

或在交付包目录（需指定主项目以加载分离版）:
  cd /data3/test-bert-modifiled/test1/bert-vits2-fused-module-delivery
  export PYTHONPATH="$(pwd)/core:$PYTHONPATH"
  export MAIN_PROJECT_DIR=/data3/test-bert-modifiled/bert-vits2-pt-modified
  USE_FUSED=0 python3 demo_inference.py
  USE_FUSED=1 python3 demo_inference.py
"""
import os
import sys
import time
import ctypes
import numpy as np

# 交付包路径
DELIVERY_DIR = os.environ.get("DELIVERY_DIR", "").strip() or os.path.dirname(os.path.abspath(__file__))
DELIVERY_CORE = os.path.join(DELIVERY_DIR, "core")
FUSED_SO = os.path.join(DELIVERY_DIR, "kernels", "fused_sdp_dp_optimized.so")
sys.path.insert(0, DELIVERY_CORE)

MAIN_DIR = os.environ.get("MAIN_PROJECT_DIR", "").strip() or os.getcwd()
SEPARATED_SO = os.environ.get("SEPARATED_SO", "").strip() or os.path.join(MAIN_DIR, "sdp_dp.so")

USE_FUSED = os.environ.get("USE_FUSED", "1") == "1"
WARMUP = 20
ITERS = 200
BATCH, SEQ_LEN, FEAT_DIM = 4, 128, 256


def run_separated(lib, audio, text):
    bs, seq_len, feat_dim = audio.shape
    sdp = np.zeros((bs, feat_dim), dtype=np.float32)
    dp = np.zeros((bs, seq_len), dtype=np.float32)
    lib.fused_forward(
        audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        text.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        sdp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        dp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        bs, seq_len, feat_dim, feat_dim,
    )
    return sdp, dp


def main():
    np.random.seed(42)
    audio = np.random.randn(BATCH, SEQ_LEN, FEAT_DIM).astype(np.float32)
    text = np.random.randn(BATCH, SEQ_LEN, FEAT_DIM).astype(np.float32)

    if USE_FUSED:
        if not os.path.exists(FUSED_SO):
            so = os.path.join(MAIN_DIR, "fused_sdp_dp_optimized.so")
            if not os.path.exists(so):
                print("ERROR: fused .so not found. Build: cd %s/kernels && bash build.sh" % DELIVERY_DIR)
                sys.exit(1)
            FUSED_SO_USE = so
        else:
            FUSED_SO_USE = FUSED_SO
        from fused_kernel_wrapper import FusedKernelWrapper
        wrapper = FusedKernelWrapper(lib_path=FUSED_SO_USE)
        def step():
            wrapper.fused_forward(audio, text)
    else:
        if not os.path.exists(SEPARATED_SO):
            print("ERROR: 分离版 sdp_dp.so 未找到。请设置 MAIN_PROJECT_DIR 或 SEPARATED_SO，或在主项目根目录运行。")
            print("  当前查找: %s" % SEPARATED_SO)
            sys.exit(1)
        lib = ctypes.CDLL(SEPARATED_SO)
        lib.fused_forward.argtypes = [
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ]
        lib.fused_forward.restype = None
        def step():
            run_separated(lib, audio, text)

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    for _ in range(WARMUP):
        step()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    t0 = time.perf_counter()
    for _ in range(ITERS):
        step()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    elapsed_ms = (time.perf_counter() - t0) * 1000

    mode = "融合" if USE_FUSED else "原版"
    print("模式: %s SDP/DP | 迭代: %d | 总耗时(ms): %.3f" % (mode, ITERS, elapsed_ms))
    print("对比: USE_FUSED=0 与 USE_FUSED=1 各跑一次，比较上面「总耗时(ms)」即可。")


if __name__ == "__main__":
    main()
