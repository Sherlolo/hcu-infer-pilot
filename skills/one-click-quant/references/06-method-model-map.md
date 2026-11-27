# 模型 ↔ 量化方法推荐表

> **何时用**：`00` 阶段 D——要按匹配度列出方案供客户选。
> **职责**：强制跑打分并展示优劣；补充 scheme/alg **语义**与导航快照。命令见 `03`/`13`，精度见 `04`。
> **权威**：
> - 首选方案排序 → `score_quant_methods.py` + [`score_rubric.yaml`](../scripts/score_rubric.yaml)（改分见 [`07`](07-match-score-rubric.md)）
> - scheme/alg 默认映射 → `llmc_oneshot.py` `_SCHEME_RECIPE`；族模板 → `quant_templates.yaml`
> - MoE-Quant 支持列表 → `registry.py`；examples → `ls examples/`（§6–7 仅为导航快照）
> **匹配度 ≠ ΔNLL**；交付仍须 `check_artifact` + `04` §3.2。

## 怎么用（客户给模型后的强制输出）

1. 完成 `00` 取证标签。
2. **必须**跑打分并按匹配度降序展示（模板见下）；不得只给单一默认方案。
3. 已适配 → 历史菜单置顶；打分结果作为辅助对照。
4. 客户确认编号 → `03`/`13` 写命令；确认前不得开跑。

```bash
python3 ./skills/scripts/score_quant_methods.py --model <HF目录> [--target fp8|int8|w4|w4a8|auto] [--size-gb N]
```

```markdown
| 排名 | 匹配度 | 方法 | 路径 | 优势 | 劣势 | 命令骨架 |
|------|--------|------|------|------|------|----------|
| 1 | … | … | … | • … | • … | … |
```

路径大类与升级阶梯以 [`00`](00-new-model-playbook.md) 总流程 / 阶段 F 为准；命令骨架以 [`03`](03-quant-runbook.md) 为准。本文不重复优先级长表。

---

## 1. llmc：算法（`--alg` / 默认推导）

> 「适合 / 风险」列含**经验倾向**（非门禁）。首选方案仍以打分为准。

| 算法 | 实现 | 要校准？ | 默认挂在哪些 scheme | 语义要点 | 硬约束 / 已知风险 |
|------|------|----------|---------------------|----------|-------------------|
| **ptq** | QuantizationModifier | 视 scheme | `FP8*`、`NVFP4A16`、`MXFP*`A16 等 | 快；weight-only 常见 | （经验）激活敏感层上可能弱于 GPTQ |
| **gptq** | GPTQModifier | **要** | `W8A16`、`W4A16` | 权重量化；**嵌套 layers + 默认 Smooth 易触发失败时**作同档修复 | 耗时/显存高；超大 Dense 易 OOM |
| **awq** | AWQModifier | **要** | `W4A16_ASYM` | 激活感知 W4 | 依赖正确 layer mapping |
| **smoothquant_gptq** | Smooth → GPTQ | **要** | `W8A8`、`W4A8`、`W4AFP8` **默认** | 标准 `model.layers.*` CausalLM | **嵌套 `language_model.layers.*`（常见 VL）** 默认 mapping 易触发失败 → `--alg gptq` |
| **smoothquant_awq** | Smooth → AWQ | **要** | 仅 `--alg` 强制 | 精度升级试验 | 同上 mapping 风险 |
| **gptq_awq** / **smoothquant_gptq_awq** | 组合 | **要** | 仅 `--alg` 强制 | 精度升级试验 | 时间≈多档相加 |
| **model_free_ptq** | 直改 safetensors | **不要** | 配合任意 scheme | 免校准、加载困难时 | **必须相对路径**（`03` §3）；YAML ignore 不生效 |
| **preprocess** | QuaRot / Flex Smooth / AWQ-SSZ | 视 transform | 非量化 | 精度升级用浮点 preprocess（`10`） | Dense 精度升级仅 QuaRot |
| **slimquant_ptq** | — | — | CLI 仍在 | — | **禁止推荐/使用** |

---

## 2. llmc：通用 scheme（语义表）

> **首选方案不以本表点名**：以匹配度 Top（客户确认）为准。

| scheme | 默认算法 | 输入权重 | 产物形态（概念） | 语义 / 常见用途 | 何时换 alg / 慎用 |
|--------|----------|----------|------------------|-----------------|-------------------|
| **FP8_DYNAMIC** | ptq | BF16/FP16 Dense | 权 per-channel FP8，激活常动态 | Dense 要 FP8、偏快/弱校准 | 要更高压缩用 W4* |
| **FP8** / **FP8_BLOCK** | ptq | 同上 | 静态 / block FP8 | 下游要对应 FP8 布局 | 确认 runtime |
| **W8A8** | smoothquant_gptq | BF16/FP16 Dense | W int8 + A int8 | Dense INT8；**默认**带 Smooth | **嵌套 `language_model.*`** → 匹配度倾向 `gptq`/`ptq`；禁止在未覆盖 `--alg` 时将默认 W8A8 作为 VL 首选方案 |
| **W8A16** | gptq | BF16/FP16 | 权 INT8、激活 FP16 | 只压权重 | — |
| **W4A16** | gptq | BF16/FP16 | 权 INT4、激活 FP16 | 强压缩 | 精度差 → AQ / ignore |
| **W4A16_ASYM** | awq | BF16/FP16 | 非对称 W4A16 | outlier 明显时 | mapping 不对会挂 |
| **W4A8** | smoothquant_gptq | BF16/FP16 | W4 + A8（**llmc 配方名**） | Dense 试验 | **≠** MoE-Quant W4A8 两档（`13` §3.1） |
| **W4AFP8** | smoothquant_gptq | BF16/FP16 | W4 + 激活 FP8 | 混合部署试验 | 依赖 runtime |
| **NVFP4 / MXFP4 / MXFP8**（及 *A16） | ptq | BF16/FP16 | 厂商/微缩放浮点 | 下游明确支持时 | 不得作为通用默认 |
| **FP8_TO_BF16** | 模板解包 | block FP8 | BF16 浮点 | 解包后再量化 | 不是「又量化一遍」 |

> `--help` scheme **不是全集**；以配方表 + YAML 为准（`03`/`11`）。

---

## 3. llmc：族专属 scheme（强绑定；须精确命中）

> 仅当 `architectures` + 权重形态（及 YAML 签名）命中时列入菜单；**禁止**按族名类推（例如「凡 Qwen3 都用 1.7B 模板」）。

| scheme | 绑定线索 | 模板 | 适合 | 不适 |
|--------|----------|------|------|------|
| **DeepseekV3_FP8_TO_INT8 / FP8** | `DeepseekV3ForCausalLM` + block FP8 | `w8a8_block_cast_channel` | 源已是 block FP8 | 源 BF16 |
| **Kimi_W4_TO_INT8 / FP8** | `KimiK25ForConditionalGeneration` + group W4 | `w4_dequant_channel` | 源已是 W4 | 普通 BF16 Dense |
| **Qwen35_MOE_BF16_TO_INT8** | `Qwen3_5MoeForConditionalGeneration` | `qwen3_5_bf16_to_channel` | 该 arch BF16 MoE | 非该 arch |
| **Qwen3 1.7B W8A8** | `Qwen3ForCausalLM` hs=2048, L=28 | `qwen3_1.7b_quant.py` | **仅**该签名 | 其它尺寸走通用 W8A8 |
| MiniMax `FP8_DYNAMIC` + ignore | `MiniMaxM2ForCausalLM` 特定签名 | 通用 + YAML ignore | 签名命中时 | 签名不对则 ignore 不生效 |

---

## 4. 按模型画像解读打分（辅助，非替代打分）

> 开跑前仍须跑 `score_quant_methods`。规模门禁以 **磁盘占用 GB（阈值 500）** 为准，不得以参数量口头分档替代。

| 模型画像 | 匹配度较高时的典型落点 | 备选 / 同档修复 | 应避免 |
|----------|--------------|-------------|----------|
| Dense BF16，标准 CausalLM，目标 FP8 | `FP8_DYNAMIC` | sequential / offload；精度差 → W8 / QuaRot | 跳过打分直接 AQ |
| Dense BF16，**非嵌套 Smooth 风险**，目标 INT8 | `W8A8`（默认 Smooth+GPTQ） | `--alg gptq`；`model_free_ptq` | — |
| Dense，**嵌套 `language_model.*`（多见 VL）**，目标 INT8 | `W8A8 --alg gptq`（或 `ptq`） | `model_free_ptq`；视觉塔 ignore 须写出模式并验收 dtype | **未改 alg 的默认 W8A8** |
| Dense 嵌套拓扑，其它目标 | 同目标 scheme + 避开默认 Smooth（打分会体现） | 同档修复见 `00` F1 | 默认 `smoothquant_*` 作为首选方案 |
| Dense 要 W4 | `W4A16` / `W4A16_ASYM` | AQ → 脚本 | llmc `W4A8` 当 MoE 档 |
| 源 block FP8 | `FP8_TO_BF16` 或 DeepSeek 族模板 | 解包后再量化 | 当 BF16 直接 GPTQ |
| Dense **>500 GB** | — | 联系管理人员 | 禁止 llmc |
| MoE + examples **强命中** | 对应 examples 脚本 | preprocess → 再脚本 / MoE-Quant | 通用 llmc 当唯一答案 |
| MoE 无强 examples | AQ 落地或 MoE-Quant（视打分） | `00` F2 | 未注册开 MoE-Quant |
| 要仅路由专家 / 混合 W4A8 | **仅** MoE-Quant（[`13` §3.1](13-ext-moe-quant.md)） | registry/脚本支持 | llmc `--scheme W4A8` |
| 已适配 | **历史菜单置顶** | 匹配度结果作辅助对照 | 隐瞒历史 |

---

## 5. examples — 导航快照（非规则）

> **非规则**：以 `ls examples/` + 各目录 README 为准；下表可能过期，只作目录检索线索。是否作为首选方案由打分 **strong/weak** 决定。

| 目录 | 脚本名线索（可能过期） |
|------|----------------------|
| `examples/deepseek/` | V4 / V4.1-Flash；W8/FP8/INT4/混合 |
| `examples/glm/` | GLM-5.3 / Flash |
| `examples/qwen/` | Qwen3.8 / Flash-Next（≠ 任意 Qwen Dense） |
| `examples/kimi/` | Kimi-K3 |
| `examples/mimo/` | MiMo2 / 2.5 / 2.6-Flash-RL |
| `examples/minimax/` | MiniMax-M3 |
| `examples/hy4/` | HY4-Preview |
| `examples/step/` | Step5 |

**打分口径**：脚本名与模型 token 重叠 → strong；仅同 family 目录 → weak（不当历史配方）。

---

## 6. MoE-Quant — 导航快照（非规则）

> **非规则**：支持列表 **只认** `registry.py` 的 `_ADAPTER_CLASSES`。下表可能过期。W4A8 术语见 [`13` §3.1](13-ext-moe-quant.md)，此处不复述。

查阅：`grep _ADAPTER_CLASSES -A30 utils/MoE-Quant/src/models/registry.py`；`ls utils/MoE-Quant/tests/run_*w4a8*.sh`。

**可走**：registry 命中；需要 EP GPTQ / 仅专家·混合 W4A8；或 F2 精度升级 / 历史菜单即 MoE-Quant。  
**不可走**：Dense；registry 未命中；把 llmc `W4A8` 说成 MoE 两档。

---

## 7. 决策摘要

```text
① score_quant_methods → 匹配度表（历史置顶）
② >500GB Dense？ → 提醒管理人员；llmc 排除
③ 客户确认 → 03/13
④ check_artifact + 04；ΔNLL≥0.1 → 00 F（客户同意精度升级）
```

## 8. 与其它文档

| 文档 | 职责 |
|------|------|
| **`06`（本文）** | 强制打分展示 + scheme 语义 + 导航快照 |
| **`07`** | 打分开发文档 |
| `score_rubric.yaml` | 分值唯一来源 |
| `00` | 总流程；D 选定首选方案；F 升级 |
| `03`/`13` | 命令；MoE 术语唯一正文 |
| `04` | ΔNLL |
| `SKILL` | 硬门禁 |
