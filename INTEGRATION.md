# 与主项目（bert-vits2-pt-modified）无缝集成指南

将本交付包（融合 SDP/DP）集成到主项目，**不修改 tn、transformers、zhmodel**，仅替换 SDP+DP 为融合算子，并支持原版 vs 融合效果对比。

---

## 一、目录关系

- **主项目**：`bert-vits2-pt-modified`（含 tn、transformers、zhmodel、README、sdp_dp.so）
- **本交付包**：`bert-vits2-fused-module-delivery`（core + kernels）

---

## 二、无缝集成（使用者必做）

### 1. 确认交付包已编译

```bash
cd bert-vits2-fused-module-delivery
cd kernels && bash build.sh && cd ..
```

### 2. 在主项目中加入“融合/原版”切换

在**主项目根目录**，找到创建或使用 DpModel/SdpModel 的代码，用下面**方式 A** 或 **方式 B**。

#### 方式 A：环境变量切换（推荐）

在最早加载 SDP/DP 的文件顶部加入（路径改成你的实际路径）：

```python
import os, sys
USE_FUSED = os.environ.get("USE_FUSED", "1") == "1"
DELIVERY_CORE = "/path/to/bert-vits2-fused-module-delivery/core"  # 改成实际路径

if USE_FUSED:
    sys.path.insert(0, DELIVERY_CORE)
    from bertvits2_fused_module import FusedSdpDpModule
    fused_module = FusedSdpDpModule(device="cuda")
    # 推理时：sdp, dp = fused_module.forward(audio_features, text_features, x_mask)
else:
    from vortex.models.bertvits2 import DpModel, SdpModel  # 或你项目实际导入
    dp_model = DpModel(...); sdp_model = SdpModel(...)
    # 推理时：sdp = sdp_model.run(audio_features); dp = dp_model.run(text_features, x_mask)
```

- **USE_FUSED=1**（默认）：用融合。
- **USE_FUSED=0**：用原版。tn、transformers 代码**不用改**。

#### 方式 B：复制本包里的切换片段

将本包内 `integration/snippet_for_main_project.py` 复制到主项目（如 `sdp_dp_switch.py`），在主项目里按该片段调用。

### 3. 不需要改的模块

- **tn**：不改。
- **transformers**：不改。
- **zhmodel**：不改。  
只改“创建/调用 SDP、DP”的那几行，按上面接入融合并支持 USE_FUSED。

---

## 三、效果对比：原版 vs 融合

### 1. 仅 SDP+DP 段耗时对比（输出加速比）

对比脚本支持环境变量 **MAIN_PROJECT_DIR**、**SEPARATED_SO**，在交付包目录下也可输出加速比。

**方式 A：在交付包目录运行（推荐）**

主项目下有 `sdp_dp.so` 时，在交付包目录执行：

```bash
cd /path/to/bert-vits2-fused-module-delivery
export PYTHONPATH="$(pwd)/core:$PYTHONPATH"
export MAIN_PROJECT_DIR=/path/to/bert-vits2-pt-modified
python3 compare_original_vs_fused.py
```

会打印：分离版 vs 融合版 各 batch 的延迟与加速比、吞吐提升%。

**方式 B：在主项目根目录运行**

```bash
cd /path/to/bert-vits2-pt-modified
export PYTHONPATH=".:/path/to/bert-vits2-fused-module-delivery/core:$PYTHONPATH"
python3 /path/to/bert-vits2-fused-module-delivery/compare_original_vs_fused.py
```

也可直接指定分离版 .so：`export SEPARATED_SO=/path/to/sdp_dp.so` 再运行脚本。未设置且当前目录无 `sdp_dp.so` 时，仅输出融合版耗时、不报错。

### 2. 全链路耗时（tn + transformers + SDP/DP）

- 原版：`USE_FUSED=0 python3 你的推理脚本.py`，记总耗时。
- 融合：`USE_FUSED=1 python3 你的推理脚本.py`，记总耗时。  
对比两次即可，tn/transformers 代码不变。
