#!/usr/bin/env python3
"""
原版 SDP+DP（分离 kernel / ONNX）vs 融合 SDP+DP 耗时对比。

用法一（在交付包目录运行，指定主项目即可得到加速比）:
  cd /path/to/bert-vits2-fused-module-delivery
  export PYTHONPATH="$(pwd)/core:$PYTHONPATH"
  export MAIN_PROJECT_DIR=/path/to/bert-vits2-pt-modified
  python3 compare_original_vs_fused.py
  # 若用 /tmp 下的脚本从别处运行，需再加: export DELIVERY_DIR=/path/to/bert-vits2-fused-module-delivery

用法二（在主项目根目录运行）:
  cd /path/to/bert-vits2-pt-modified
  export PYTHONPATH=".:/path/to/bert-vits2-fused-module-delivery/core:$PYTHONPATH"
  python3 /path/to/bert-vits2-fused-module-delivery/compare_original_vs_fused.py

也可直接指定分离版 .so:  export SEPARATED_SO=/path/to/sdp_dp.so
"""
import os
import sys
import time
import ctypes
import numpy as np

# 交付包路径：优先环境变量（从任意目录运行时可指定），否则为本脚本所在目录
DELIVERY_DIR = os.environ.get("DELIVERY_DIR", "").strip() or os.path.dirname(os.path.abspath(__file__))
DELIVERY_CORE = os.path.join(DELIVERY_DIR, "core")
FUSED_SO = os.path.join(DELIVERY_DIR, "kernels", "fused_sdp_dp_optimized.so")
sys.path.insert(0, DELIVERY_CORE)

# 主项目目录与分离版 .so：支持环境变量，便于在交付包目录下也能输出加速比
MAIN_DIR = os.environ.get("MAIN_PROJECT_DIR", "").strip() or os.getcwd()
SEPARATED_SO = os.environ.get("SEPARATED_SO", "").strip() or os.path.join(MAIN_DIR, "sdp_dp.so")

def load_separated():
    if not os.path.exists(SEPARATED_SO):
        return None
    import ctypes
    lib = ctypes.CDLL(SEPARATED_SO)
    lib.fused_forward.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
    ]
    lib.fused_forward.restype = None
    return lib

def run_separated(lib, audio, text):
    bs, seq_len, feat_dim = audio.shape
    audio = audio.astype(np.float32)
    text = text.astype(np.float32)
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

def load_fused():
    fused_so = FUSED_SO
    if not os.path.exists(fused_so):
        main_fused = os.path.join(MAIN_DIR, "fused_sdp_dp_optimized.so")
        if os.path.exists(main_fused):
            fused_so = main_fused
        else:
            raise FileNotFoundError(
                "Fused .so not found. Either:\n"
                "  1) cd %s/kernels && bash build.sh\n"
                "  2) Or copy fused_sdp_dp_optimized.so to main project root: %s"
                % (DELIVERY_DIR, MAIN_DIR)
            )
    from fused_kernel_wrapper import FusedKernelWrapper
    return FusedKernelWrapper(lib_path=fused_so)

def run_fused(wrapper, audio, text):
    return wrapper.fused_forward(audio.astype(np.float32), text.astype(np.float32))

def benchmark(fn, audio, text, warmup=50, iterations=500):
    for _ in range(warmup):
        fn(audio, text)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    times = []
    for _ in range(iterations):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        t0 = time.perf_counter()
        fn(audio, text)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    median_ms = float(np.median(times))
    mean_ms = float(np.mean(times))
    return {
        "mean": mean_ms,
        "median": median_ms,
        "std": float(np.std(times)),
        "throughput": len(audio) * 1000 / mean_ms,
        "throughput_median": len(audio) * 1000 / median_ms,
    }

def main():
    print("=" * 70)
    print("原版 SDP+DP vs 融合 SDP+DP 耗时对比")
    print("=" * 70)
    print("主项目目录: %s" % MAIN_DIR)
    print("交付包目录: %s" % DELIVERY_DIR)
    print("分离版 .so: %s (存在=%s)" % (SEPARATED_SO, os.path.exists(SEPARATED_SO)))
    print("融合版 .so: %s (存在=%s)" % (FUSED_SO, os.path.exists(FUSED_SO)))
    if not os.path.exists(SEPARATED_SO):
        print()
        print("提示：未找到 分离版 sdp_dp.so，仅输出融合版耗时。")
        print("要得到加速比，请设置: export MAIN_PROJECT_DIR=/path/to/bert-vits2-pt-modified")
        print("或: export SEPARATED_SO=/path/to/sdp_dp.so")
    print()

    configs = [
        {"batch_size": 1, "seq_len": 128, "feat_dim": 256},
        {"batch_size": 4, "seq_len": 128, "feat_dim": 256},
        {"batch_size": 8, "seq_len": 128, "feat_dim": 256},
        {"batch_size": 32, "seq_len": 128, "feat_dim": 256},
    ]

    separated_lib = load_separated()
    fused_wrapper = load_fused()

    print("warmup=50, iterations=500，Speedup 按中位数计算（抗离群值）")
    print("%-8s %22s %22s %10s %12s" % ("Batch", "分离版(ms)", "融合版(ms)", "Speedup", "吞吐提升%"))
    print("-" * 76)

    for cfg in configs:
        bs, seq_len, feat_dim = cfg["batch_size"], cfg["seq_len"], cfg["feat_dim"]
        np.random.seed(42)
        audio = np.random.randn(bs, seq_len, feat_dim).astype(np.float32)
        text = np.random.randn(bs, seq_len, feat_dim).astype(np.float32)

        sep_stats = None
        if separated_lib:
            def sep_fn(a, t):
                return run_separated(separated_lib, a, t)
            sep_stats = benchmark(sep_fn, audio, text)

        def fused_fn(a, t):
            return run_fused(fused_wrapper, a, t)
        fus_stats = benchmark(fused_fn, audio, text)

        speedup = (sep_stats["median"] / fus_stats["median"]) if sep_stats else None
        gain = ((fus_stats["throughput_median"] - sep_stats["throughput_median"]) / sep_stats["throughput_median"] * 100) if sep_stats else None
        sep_str = "median %.4f ± %.4f" % (sep_stats["median"], sep_stats["std"]) if sep_stats else "-"
        fus_str = "median %.4f ± %.4f" % (fus_stats["median"], fus_stats["std"])
        print("%-8s %22s %22s %10s %12s" % (
            bs, sep_str, fus_str,
            "%.2fx" % speedup if speedup else "-",
            "%.1f%%" % gain if gain is not None else "-",
        ))

    print("=" * 76)
    print("说明：融合版使用交付包 kernels/fused_sdp_dp_optimized.so（设备内存缓存）。")
    print("batch=1 时分离版无缓存、方差大，用中位数与 500 次迭代可得到更稳定的 Speedup。")
    print("全链路对比：USE_FUSED=0 与 USE_FUSED=1 分别跑你的推理脚本对比总耗时。")
    print("=" * 76)

if __name__ == "__main__":
    main()
