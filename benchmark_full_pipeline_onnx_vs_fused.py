#!/usr/bin/env python3
"""
全链路耗时对比:未融合:ONNX SDP+DP)vs 融合(fused_sdp_dp_optimized.so),记录总耗时。

【重要】本脚本的「全链路」= 仅「编码器输出 → SDP/DP」这一段,不含 tn、也不含编码器的真实执行。
  - tn:交付包内只有 tn/*.fst 资源,无 Python 调用;tn 由主项目执行,本脚本不跑。
  - 编码器:本脚本不跑真实编码器(transformers/zhmodel),仅用随机 x 或 ENCODER_SIM_MS 模拟「编码器输出」。
  - 因此:本脚本对比的是「从编码器输出到 SDP/DP 结束」的总耗时,不是「tn + 编码器 + SDP/DP」的完整推理全流程。

若要做「含 tn + 编码器」的真正全流程耗时对比,请在主项目里:
  USE_FUSED=0 python3 你的推理脚本.py   # 记录总耗时
  USE_FUSED=1 python3 你的推理脚本.py   # 记录总耗时
  对比两次总耗时即可(见《完整使用文档》6.4)。

本脚本流程(仅 SDP/DP 段):
  - 原版:ONNX BertVits2.2PT_sdp.onnx + BertVits2.2PT_dp.onnx 分别推理
  - 融合:fused_sdp_dp_optimized.so 一次调用

可选环境变量 ENCODER_SIM_MS:模拟编码器耗时(毫秒),加入每轮总耗时,默认 0。

用法(在交付包根目录执行,且交付包下含 zhmodel/.../BertVits2.2PT/):
  cd /path/to/bert-vits2-fused-module-delivery
  export PYTHONPATH="$(pwd)/core:$PYTHONPATH"
  python3 benchmark_full_pipeline_onnx_vs_fused.py

若本脚本放在项目总结下,需指定交付包路径后执行:
  export DELIVERY_DIR=/path/to/bert-vits2-fused-module-delivery
  export PYTHONPATH="$DELIVERY_DIR/core:$PYTHONPATH"
  python3 benchmark_full_pipeline_onnx_vs_fused.py
"""
import os
import sys
import time
import numpy as np

# 交付包路径:优先环境变量,否则按本脚本所在目录推断
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DELIVERY_DIR = os.environ.get("DELIVERY_DIR", "").strip() or os.environ.get("BERT_VITS2_FUSED_DELIVERY", "").strip()
if not DELIVERY_DIR:
    _cand = os.path.join(_SCRIPT_DIR, "..", "test1", "bert-vits2-fused-module-delivery")
    if os.path.isdir(_cand) and os.path.isfile(os.path.join(_cand, "benchmark_onnx_vs_fused.py")):
        DELIVERY_DIR = os.path.abspath(_cand)
if not DELIVERY_DIR or not os.path.isdir(DELIVERY_DIR):
    print("未找到交付包目录,请设置: export DELIVERY_DIR=/path/to/bert-vits2-fused-module-delivery")
    sys.exit(1)

DELIVERY_CORE = os.path.join(DELIVERY_DIR, "core")
if DELIVERY_CORE not in sys.path:
    sys.path.insert(0, DELIVERY_CORE)


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

WARMUP = 20
ITERS = 200
ONNX_CHANNELS = 192
ONNX_EMB_DIM = 256
ENCODER_SIM_MS = float(os.environ.get("ENCODER_SIM_MS", "0"))


def _make_onnx_inputs(batch, seq_len):
    np.random.seed(42)
    x = np.random.randn(batch, ONNX_CHANNELS, seq_len).astype(np.float32)
    x_mask = np.ones((batch, 1, seq_len), dtype=np.float32)
    zin = np.random.randn(batch, 2, seq_len).astype(np.float32)
    g = np.random.randn(1, ONNX_EMB_DIM, 1).astype(np.float32)
    return x, x_mask, zin, g


def _encoder_sim():
    if ENCODER_SIM_MS > 0:
        time.sleep(ENCODER_SIM_MS / 1000.0)


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


def _sync_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass


def benchmark_full_pipeline_onnx(sess_sdp, sess_dp, x, x_mask, zin, g, batch, warmup=WARMUP, iterations=ITERS):
    """全链路:encoder_sim + ONNX SDP + ONNX DP,记录每轮总耗时(ms)。"""
    for _ in range(warmup):
        _encoder_sim()
        run_onnx_sdp(sess_sdp, x, x_mask, zin, g)
        run_onnx_dp(sess_dp, x, x_mask, g)
    _sync_cuda()
    times = []
    for _ in range(iterations):
        _sync_cuda()
        t0 = time.perf_counter()
        _encoder_sim()
        run_onnx_sdp(sess_sdp, x, x_mask, zin, g)
        run_onnx_dp(sess_dp, x, x_mask, g)
        _sync_cuda()
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {
        "median": float(np.median(times)),
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
    }


def benchmark_full_pipeline_fused(wrapper, audio, text, batch, warmup=WARMUP, iterations=ITERS):
    """全链路:encoder_sim + fused_forward,记录每轮总耗时(ms)。"""
    for _ in range(warmup):
        _encoder_sim()
        run_fused(wrapper, audio, text)
    _sync_cuda()
    times = []
    for _ in range(iterations):
        _sync_cuda()
        t0 = time.perf_counter()
        _encoder_sim()
        run_fused(wrapper, audio, text)
        _sync_cuda()
        times.append((time.perf_counter() - t0) * 1000)
    times = np.array(times)
    return {
        "median": float(np.median(times)),
        "mean": float(np.mean(times)),
        "std": float(np.std(times)),
    }


def main():
    print("=" * 72)
    print("全链路耗时对比:未融合(ONNX SDP+DP) vs 融合(fused_sdp_dp_optimized.so)")
    print("=" * 72)
    print("说明: 本脚本仅计时「编码器输出 → SDP/DP」段,不含 tn、不含编码器真实执行。")
    print("      含 tn+编码器的真正全流程对比需在主项目用 USE_FUSED=0/1 跑完整推理(见文档 6.4)。")
    print("流程: 模拟编码器输出 x 后 → SDP+DP 段(原版两段 ONNX / 融合一段 .so)")
    print("交付包: %s" % DELIVERY_DIR)
    print("ONNX 目录: %s" % ONNX_DIR)
    print("SDP ONNX:  %s (存在=%s)" % (SDP_ONNX, os.path.isfile(SDP_ONNX)))
    print("DP ONNX:   %s (存在=%s)" % (DP_ONNX, os.path.isfile(DP_ONNX)))
    print("融合 .so:  %s (存在=%s)" % (FUSED_SO, os.path.isfile(FUSED_SO)))
    if ENCODER_SIM_MS > 0:
        print("编码器模拟: 每轮 %.2f ms (测的是 模拟编码器+SDP+DP 总耗时)" % ENCODER_SIM_MS)
    else:
        print("编码器模拟: 关闭 (纯 SDP+DP)")
    print("warmup=%d, iterations=%d,总耗时按中位数(ms)" % (WARMUP, ITERS))
    print("-" * 72)

    if not os.path.isfile(SDP_ONNX) or not os.path.isfile(DP_ONNX):
        print("请确保交付包下 zhmodel 含有 BertVits2.2PT_sdp.onnx 与 BertVits2.2PT_dp.onnx,或设置 ZHMODEL_ONNX_DIR")
        sys.exit(1)
    if not os.path.isfile(FUSED_SO):
        print("请先编译: cd %s/kernels && bash build.sh" % DELIVERY_DIR)
        sys.exit(1)

    sess_sdp, sess_dp = load_onnx_sdp_dp()
    fused_wrapper = load_fused()

    configs = [
        {"batch_size": 1, "seq_len": 128},
        {"batch_size": 4, "seq_len": 128},
        {"batch_size": 8, "seq_len": 128},
        {"batch_size": 32, "seq_len": 128},
    ]

    print("%-8s %26s %26s %10s" % ("Batch", "原版(ONNX)全链路(ms)", "融合全链路(ms)", "Speedup"))
    print("-" * 72)

    for cfg in configs:
        bs, seq_len = cfg["batch_size"], cfg["seq_len"]
        x, x_mask, zin, g = _make_onnx_inputs(bs, seq_len)
        audio = np.transpose(x, (0, 2, 1)).copy()
        text = np.transpose(x, (0, 2, 1)).copy()

        try:
            onnx_stats = benchmark_full_pipeline_onnx(sess_sdp, sess_dp, x, x_mask, zin, g, bs, iterations=ITERS)
        except Exception as e:
            print("%-8s %26s %26s %10s" % (bs, "ERROR: %s" % str(e)[:18], "-", "-"))
            continue
        try:
            fus_stats = benchmark_full_pipeline_fused(fused_wrapper, audio, text, bs, iterations=ITERS)
        except Exception as e:
            print("%-8s %26s %26s %10s" % (bs, "median %.2f" % onnx_stats["median"], "ERROR: %s" % str(e)[:18], "-"))
            continue

        speedup = onnx_stats["median"] / fus_stats["median"]
        onnx_str = "median %.2f ± %.2f" % (onnx_stats["median"], onnx_stats["std"])
        fus_str = "median %.2f ± %.2f" % (fus_stats["median"], fus_stats["std"])
        print("%-8s %26s %26s %10s" % (bs, onnx_str, fus_str, "%.2fx" % speedup))

    print("=" * 72)
    print("说明:原版 = BertVits2.2PT_sdp.onnx + BertVits2.2PT_dp.onnx 分别推理；融合 = fused_sdp_dp_optimized.so 一次调用。")
    print("      本脚本不含 tn/编码器；可选 ENCODER_SIM_MS=2.0 模拟编码器耗时。")
    print("=" * 72)


if __name__ == "__main__":
    main()
