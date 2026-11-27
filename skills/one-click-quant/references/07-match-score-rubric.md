# 匹配度打分规则（开发文档）

> **何时用**：要改匹配度因子分值、理解 `score_quant_methods.py` 怎么打分、或排查「为什么这个方法分高/低」。
> **数字唯一来源**：[`skills/scripts/score_rubric.yaml`](../scripts/score_rubric.yaml)（**每条分值旁注释即得分理由**）  
> **执行实现**：[`skills/scripts/score_quant_methods.py`](../scripts/score_quant_methods.py)  
> **对客户怎么用**：见 [`06-method-model-map.md`](06-method-model-map.md)（推荐流程）；本文不重复方法对照表。
>
> 匹配度 = **工程可行性**（0–100），**不是** ΔNLL / 精度预测。

## 1. 改分流程（必读）

1. **只改** `skills/scripts/score_rubric.yaml` 里的数字（或 `gates`），不要在 `.py` 里写死分值；**改数字时同步改同行/邻行注释里的得分理由**。
2. 若新增 / 重命名 key：同步改 `score_quant_methods.py` 里对应 `_F(...)` / `_B(...)` 调用，并更新本文 §3。
3. 冒烟（仓库根）：

```bash
# 最小：rubric 能加载 +「嵌套 VL 拓扑」冒烟（用含 vision_config 的 stub，不绑定某一发版模型名）
TMP=$(mktemp -d)
cat > "$TMP/config.json" <<'EOF'
{"model_type":"qwen2_5_vl","architectures":["Qwen2_5_VLForConditionalGeneration"],"vision_config":{}}
EOF
python3 ./skills/scripts/score_quant_methods.py \
  --model "$TMP" --size-gb 140 --target int8 --top 5
rm -rf "$TMP"

# 可选：指定其它 rubric 文件做 A/B
python3 ./skills/scripts/score_quant_methods.py \
  --model <真实HF目录> --rubric ./skills/scripts/score_rubric.yaml --format json | head
```

4. 预期（当前默认 rubric）：`is_vl=true` + `--target int8` 时，带 `--alg gptq|ptq` 的 W8A8 应高于**未改 alg** 的默认 SmoothQuant W8A8（`vl.smooth_default_penalty`）。条件是「脚本判为 VL/嵌套风险」，不是某一个具体权重目录。
5. 同步双树：改完后把 `score_rubric.yaml` / `score_quant_methods.py` / 本文拷到 `.cursor/skills/one-click-quant/` 对应路径。

## 2. 总分公式

```text
score = clamp(score_clamp.min, score_clamp.max,  base + Σ factors)
```

- `base`：来自 `llmc_base.*`，或 examples / MoE-Quant / autoquantize 段内的 base 组合。
- 硬门禁：不进入累加，直接 `excluded`、分数 0（见 §4）。
- 每个因子会写入输出里的 `breakdown`，并生成 `pros` / `cons` 文案（文案在 `.py`，分值在 `.yaml`）。

## 3. Key 一览（与 yaml 对齐）

### 3.1 `score_clamp` / `gates`

| Key | 含义 |
|-----|------|
| `score_clamp.min` / `max` | 最终分数上下限（默认 0–100） |
| `gates.dense_ultra_gb` | Dense 且 `size_gb` 大于该值 → **排除全部 llmc** |

### 3.2 `llmc_base.*`（llmc 候选起点）

| Key | 对应方法 |
|-----|----------|
| `FP8_DYNAMIC` | `--scheme FP8_DYNAMIC` |
| `W8A8_default` | `W8A8` 且未指定 `--alg`（默认 smoothquant_gptq） |
| `W8A8_gptq` / `W8A8_ptq` / `W8A8_model_free_ptq` | 显式 `--alg` |
| `W4A16` / `W4A16_ASYM` / `W4A8` | 对应 scheme |
| `FP8_TO_BF16` | 解包（kind=`unpack`） |
| `family_scheme` | `quant_templates.yaml` 族专属 scheme 的 base |

### 3.3 `factors.*`

| Key | 何时生效 |
|-----|----------|
| `topology.dense_llmc` | Dense → llmc |
| `topology.moe_llmc` | MoE → 通用 llmc（通常为负） |
| `topology.moe_examples_strong` | MoE + examples 强命中 |
| `topology.dense_examples_strong` | Dense + examples 强命中 |
| `weight.bf16_generic_match` | 源 bf16/fp16 且 scheme 为通用量化 kind |
| `weight.block_fp8_unpack_match` | 源 block FP8 且 kind=`unpack` |
| `weight.block_fp8_generic_mismatch` | 源 block FP8 却走非解包通用 scheme |
| `weight.w4_as_bf16_scheme` | 源已是 W4 却套 int8/fp8 通用 scheme |
| `size.llmc_within_gate` | `size_gb ≤ gates.dense_ultra_gb` 且走 llmc |
| `vl.smooth_default_penalty` | VL + W8A8/W4A8/W4AFP8 且 alg 未覆盖 |
| `vl.gptq_bonus` / `ptq_bonus` / `model_free_bonus` | VL + 对应 `--alg` |
| `history.llmc_side_evidence` | `--has-history` 时其它 llmc 辅助对照 |
| `family.weight_mismatch` | 族模板与源权重矛盾（DeepSeek FP8 / Kimi W4） |
| `family.yaml_hit` | arch 命中 YAML 族 scheme |
| `examples.strong_base` + `strong_hit` | 强命中起始分二者之和 |
| `examples.weak_base` + `weak_family_only` | 弱命中（仅同 family）起始分 |
| `examples.history_boost` | 强命中且 `--has-history` |
| `autoquantize.base` / `moe_adapter_bonus` | MoE 无强 examples 时的升级候选 |
| `moe_quant.base` + `registry` | MoE-Quant 起始分 |
| `moe_quant.run_script` | 存在 `tests/run_*w4a8*.sh` |
| `moe_quant.non_moe_topology` | 非 MoE 却进 MoE-Quant（应大幅为负） |

### 3.4 `target_bonus`

| Key | 含义 |
|-----|------|
| `auto_neutral` | `--target auto` 时所有 kind 的统一加分 |
| `fp8` / `int8` / `w4` / `w4a8` | 客户目标 → 与候选 `kind`（`fp8`/`int8`/`w4`/`w4a8`/`unpack`/`mixed`）对照表 |

## 4. 硬门禁（逻辑在脚本，阈值阈值多在 yaml）

| 条件 | 结果 |
|------|------|
| Dense 且 `size_gb > gates.dense_ultra_gb` | 全部 llmc `excluded` |
| MoE-Quant registry 未命中 | MoE-Quant 整类排除 |
| `slimquant_ptq` | 永远排除 |

## 5. 取证标签（脚本推断，非 yaml）

打分前由 `score_quant_methods.py` 从 `config.json` / 权重线索生成：`topology`、`is_vl`、`weight`、`size_gb`、`examples_hits`（strong/weak）、`moe_adapter`、`yaml_schemes`。可用 CLI 覆盖：`--size-gb`、`--weight`、`--has-history`、`--target`。

## 6. 与其它文档

| 文档 | 职责 |
|------|------|
| **`07`（本文）** | 开发：改分、key 含义、冒烟 |
| `score_rubric.yaml` | **分值唯一来源** |
| `06` | 客户侧：方法对照 + 强制按匹配度列表 |
| `00` 阶段 D | 取证后必须跑打分并展示 |
| `04` | 精度 ΔNLL；与匹配度无关 |
