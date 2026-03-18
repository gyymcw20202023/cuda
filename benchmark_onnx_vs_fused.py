#!/usr/bin/env python3
"""
原始 ONNX 推理（SDP + DP 分别跑）vs 融合 CUDA（fused_sdp_dp_optimized.so）性能对比。

不使用 sdp_dp.so，仅对比：
  - 原版：zhmodel 下 BertVits2.2PT_sdp.onnx + BertVits2.2PT_dp.onnx 分别推理
  - 融合版：kernels/fused_sdp_dp_optimized.so 一次调用

用法（在交付包根目录执行，且交付包下含 zhmodel/onnx_models_ziyan/.../BertVits2.2PT/）:
  cd /data3/test-bert-modifiled/test1/bert-vits2-fused-module-delivery
  export PYTHONPATH="$(pwd)/core:$PYTHONPATH"
  pip install onnxruntime-gpu  # 或 onnxruntime
  python3 benchmark_onnx_vs_fused.py

可通过环境变量指定 ONNX 目录: ZHMODEL_ONNX_DIR=/path/to/BertVits2.2PT
"""
import os
import sys
import time
import numpy as np

DELIVERY_DIR = os.path.dirname(os.path.abspath(__file__))
DELIVERY_CORE = os.path.join(DELIVERY_DIR, "core")
sys.path.insert(0, DELIVERY_CORE)

# zhmodel 下 ONNX 目录
def _default_onnx_dir():
    candidates = [
        os.path.join(DELIVERY_DIR, "zhmodel", "onnx_models_ziyan", "BR-TTS-CF-DongNan-330K-20250902", "BertVits2.2PT"),
        os.path.join(DELIVERY_DIR, "zhmodel", "onnx_base"),
    ]
    for d in candidates:
        sdp = os.path.join(d, "BertVits2.2PT_sdp.onnx")
        dp = os.path.join(d, "BertVits2.2PT_dp.onnx")
        if os.path.isfile(sdp) and os.path.isfile(dp):
            return d
    return candidates[0]

ONNX_DIR = os.environ.get("ZHMODEL_ONNX_DIR", "").strip() or _default_onnx_dir()
SDP_ONNX = os.path.join(ONNX_DIR, "BertVits2.2PT_sdp.onnx")
DP_ONNX = os.path.join(ONNX_DIR, "BertVits2.2PT_dp.onnx")
FUSED_SO = os.path.join(DELIVERY_DIR, "kernels", "fused_sdp_dp_optimized.so")

WARMUP = 30
ITERS = 300


# zhmodel ONNX 实际输入：x [batch, 192, seq_len], x_mask [batch, 1, seq_len], zin [batch, 2, seq_len](SDP), g [1, 256, 1]
ONNX_CHANNELS = 192
ONNX_EMB_DIM = 256


def _make_onnx_inputs(batch, seq_len):
    """生成 ONNX SDP/DP 所需输入：x (b,192,s), x_mask (b,1,s), zin (b,2,s), g (1,256,1)。"""
    np.random.seed(42)
    x = np.random.randn(batch, ONNX_CHANNELS, seq_len).astype(np.float32)
    x_mask = np.ones((batch, 1, seq_len), dtype=np.float32)
    zin = np.random.randn(batch, 2, seq_len).astype(np.float32)
    g = np.random.randn(1, ONNX_EMB_DIM, 1).astype(np.float32)
    return x, x_mask, zin, g


def load_onnx_sdp_dp():
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError("需要 onnxruntime: pip install onnxruntime 或 onnxruntime-gpu")
    if not os.path.isfile(SDP_ONNX):
        raise FileNotFoundError("SDP ONNX 不存在: %s" % SDP_ONNX)
    if not os.path.isfile(DP_ONNX):
        raise FileNotFoundError("DP ONNX 不存在: %s" % DP_ONNX)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if "CUDAExecutionProvider" in ort.get_available_providers() else ["CPUExecutionProvider"]
    sess_sdp = ort.InferenceSession(SDP_ONNX, opts, providers=providers)
    sess_dp = ort.InferenceSession(DP_ONNX, opts, providers=providers)
    return sess_sdp, sess_dp


def run_onnx_sdp(sess_sdp, x, x_mask, zin, g):
    feeds = {"x": x, "x_mask": x_mask, "zin": zin, "g": g}
    out_names = [o.name for o in sess_sdp.get_outputs()]
    return sess_sdp.run(out_names, feeds)


def run_onnx_dp(sess_dp, x, x_mask, g):
    feeds = {"x": x, "x_mask": x_mask, "g": g}
    out_names = [o.name for o in sess_dp.get_outputs()]
    return sess_dp.run(out_names, feeds)


def load_fused():
    if not os.path.isfile(FUSED_SO):
        raise FileNotFoundError("融合 .so 不存在: %s\n请执行: cd %s/kernels && bash build.sh" % (FUSED_SO, DELIVERY_DIR))
    from fused_kernel_wrapper import FusedKernelWrapper
    return FusedKernelWrapper(lib_path=FUSED_SO)


def run_fused(wrapper, audio, text):
    return wrapper.fused_forward(audio.astype(np.float32), text.astype(np.float32))


def benchmark_onnx(sess_sdp, sess_dp, x, x_mask, zin, g, batch, warmup=WARMUP, iterations=ITERS):
    for _ in range(warmup):
        run_onnx_sdp(sess_sdp, x, x_mask, zin, g)
        run_onnx_dp(sess_dp, x, x_mask, g)
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
        run_onnx_sdp(sess_sdp, x, x_mask, zin, g)
        run_onnx_dp(sess_dp, x, x_mask, g)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {
        "median": float(np.median(times)),
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "throughput_median": batch * 1000 / float(np.median(times)),
    }


def benchmark_fused(wrapper, audio, text, batch, warmup=WARMUP, iterations=ITERS):
    for _ in range(warmup):
        run_fused(wrapper, audio, text)
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
        run_fused(wrapper, audio, text)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {
        "median": float(np.median(times)),
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
        "throughput_median": batch * 1000 / float(np.median(times)),
    }


def main():
    print("=" * 72)
    print("原始 ONNX（SDP+DP 分别推理）vs 融合 CUDA（fused_sdp_dp_optimized.so）性能对比")
    print("=" * 72)
    print("ONNX 目录: %s" % ONNX_DIR)
    print("SDP ONNX:  %s (存在=%s)" % (SDP_ONNX, os.path.isfile(SDP_ONNX)))
    print("DP ONNX:   %s (存在=%s)" % (DP_ONNX, os.path.isfile(DP_ONNX)))
    print("融合 .so:  %s (存在=%s)" % (FUSED_SO, os.path.isfile(FUSED_SO)))
    if not os.path.isfile(SDP_ONNX) or not os.path.isfile(DP_ONNX):
        print()
        print("请确保交付包下 zhmodel 含有 BertVits2.2PT_sdp.onnx 与 BertVits2.2PT_dp.onnx，")
        print("或设置: export ZHMODEL_ONNX_DIR=/path/to/BertVits2.2PT")
        sys.exit(1)
    if not os.path.isfile(FUSED_SO):
        print("\n请先编译融合 kernel: cd %s/kernels && bash build.sh" % DELIVERY_DIR)
        sys.exit(1)
    print()

    sess_sdp, sess_dp = load_onnx_sdp_dp()
    fused_wrapper = load_fused()

    # ONNX 使用 x [batch, 192, seq_len]；融合 kernel 使用 [batch, seq_len, feat_dim]，feat_dim=192
    configs = [
        {"batch_size": 1, "seq_len": 128},
        {"batch_size": 4, "seq_len": 128},
        {"batch_size": 8, "seq_len": 128},
        {"batch_size": 32, "seq_len": 128},
    ]

    print("warmup=%d, iterations=%d，Speedup 按中位数计算（ONNX 输入 x [batch,192,seq_len]）" % (WARMUP, ITERS))
    print("%-8s %24s %24s %10s %12s" % ("Batch", "ONNX(ms)", "融合(ms)", "Speedup", "吞吐提升%"))
    print("-" * 76)

    for cfg in configs:
        bs, seq_len = cfg["batch_size"], cfg["seq_len"]
        x, x_mask, zin, g = _make_onnx_inputs(bs, seq_len)
        # 融合 kernel 输入: [batch, seq_len, feat_dim]，与 ONNX x (b,192,s) 对应为 (b,s,192)
        audio = np.transpose(x, (0, 2, 1)).copy()   # (b, s, 192)
        text = np.transpose(x, (0, 2, 1)).copy()   # (b, s, 192)

        try:
            onnx_stats = benchmark_onnx(sess_sdp, sess_dp, x, x_mask, zin, g, bs, iterations=ITERS)
        except Exception as e:
            print("%-8s %24s %24s %10s %12s" % (bs, "ERROR: %s" % str(e)[:20], "-", "-", "-"))
            continue
        fus_stats = benchmark_fused(fused_wrapper, audio, text, bs, iterations=ITERS)

        speedup = onnx_stats["median"] / fus_stats["median"]
        gain = (fus_stats["throughput_median"] - onnx_stats["throughput_median"]) / onnx_stats["throughput_median"] * 100
        onnx_str = "median %.4f ± %.4f" % (onnx_stats["median"], onnx_stats["std"])
        fus_str = "median %.4f ± %.4f" % (fus_stats["median"], fus_stats["std"])
        print("%-8s %24s %24s %10s %12s" % (bs, onnx_str, fus_str, "%.2fx" % speedup, "%.1f%%" % gain))

    print("=" * 76)
    print("说明：ONNX = BertVits2.2PT_sdp.onnx + BertVits2.2PT_dp.onnx 分别推理；融合 = fused_sdp_dp_optimized.so 一次调用。")
    print("=" * 76)


if __name__ == "__main__":
    main()
