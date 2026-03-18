#!/usr/bin/env python3
"""
融合 SDP/DP 真实接入示例（单文件、可独立运行）。

模拟主项目流程：tn → 编码器(transformers/zhmodel) 得到 x_audio、x_text [b,192,s]
→ 本包 FusedSdpDpModule.forward(...) → sdp_embedding、dp_durations。

运行方式（任选其一）：
  export DELIVERY_DIR=/path/to/bert-vits2-fused-module-delivery
  python3 quick_integration_example_真实接入.py

  cd /path/to/bert-vits2-fused-module-delivery && PYTHONPATH="$(pwd)/core:$PYTHONPATH" python3 /path/to/项目总结/quick_integration_example_真实接入.py
"""
import os
import sys

def _find_delivery_core():
    """优先用环境变量，否则按本脚本位置推断交付包路径。"""
    root = os.path.abspath(os.path.dirname(__file__))
    # 项目总结/ 下时，交付包在 ../../test1/bert-vits2-fused-module-delivery
    for base in [
        os.environ.get("DELIVERY_DIR", ""),
        os.environ.get("BERT_VITS2_FUSED_DELIVERY", ""),
        os.path.join(root, "..", "test1", "bert-vits2-fused-module-delivery"),
        os.path.join(root, "..", "..", "test1", "bert-vits2-fused-module-delivery"),
    ]:
        if not base:
            continue
        base = os.path.abspath(base)
        core = os.path.join(base, "core")
        if os.path.isdir(core) and os.path.isfile(os.path.join(core, "bertvits2_fused_module.py")):
            return core, base
    return None, None

CORE_DIR, DELIVERY_DIR = _find_delivery_core()
if CORE_DIR is None:
    print("未找到交付包 core 目录，请设置: export DELIVERY_DIR=/path/to/bert-vits2-fused-module-delivery")
    sys.exit(1)
if CORE_DIR not in sys.path:
    sys.path.insert(0, CORE_DIR)

import torch
from bertvits2_fused_module import FusedSdpDpModule


def run_real_integration_test():
    """
    最真实示例：编码器输出形状与 zhmodel ONNX 一致 [batch, 192, seq_len]，
    转成融合所需 [batch, seq_len, 192] 后一次 forward。
    """
    # 与 zhmodel/BertVits2.2PT_sdp.onnx、dp.onnx 输入一致：x [b, 192, s]
    BATCH, CHANNELS, SEQ_LEN = 1, 192, 128
    feat_dim = CHANNELS

    # 模拟 tn → 编码器 后的两路输出（主项目里来自 transformers/zhmodel）
    x_audio = torch.randn(BATCH, CHANNELS, SEQ_LEN, device="cuda", dtype=torch.float32)
    x_text = torch.randn(BATCH, CHANNELS, SEQ_LEN, device="cuda", dtype=torch.float32)
    x_mask = torch.ones(BATCH, 1, SEQ_LEN, device="cuda", dtype=torch.float32)

    # 形状转换：(b, 192, s) → (b, s, 192)，与 benchmark_onnx_vs_fused 一致
    audio_features = x_audio.permute(0, 2, 1).contiguous()
    text_features = x_text.permute(0, 2, 1).contiguous()

    # 融合模块（主项目里只初始化一次）
    fused_module = FusedSdpDpModule(device="cuda")

    # 真实接入：一段替代原两段 sdp_model.run(...) + dp_model.run(...)
    sdp_embedding, dp_durations = fused_module.forward(audio_features, text_features, x_mask)

    # 校验形状
    assert sdp_embedding.shape == (BATCH, feat_dim), sdp_embedding.shape
    assert dp_durations.shape == (BATCH, SEQ_LEN), dp_durations.shape

    return {
        "sdp_shape": tuple(sdp_embedding.shape),
        "dp_shape": tuple(dp_durations.shape),
        "sdp_dtype": str(sdp_embedding.dtype),
        "dp_dtype": str(dp_durations.dtype),
    }


def main():
    print("融合 SDP/DP 真实接入示例（单文件）")
    print("交付包 core:", CORE_DIR)
    print("-" * 50)

    try:
        out = run_real_integration_test()
        print("SDP 输出 shape:", out["sdp_shape"], "dtype:", out["sdp_dtype"])
        print("DP 输出 shape:", out["dp_shape"], "dtype:", out["dp_dtype"])
        print("-" * 50)
        print("OK 真实接入测试通过")
        return 0
    except Exception as e:
        print("FAIL:", e)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
