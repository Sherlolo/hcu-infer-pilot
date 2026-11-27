# AutoQuantize：混合精度位宽计划（`utils/auto_quantize`）

> **权威来源**：`utils/auto_quantize/README.md`、`python -m utils.auto_quantize --help`、
> `utils/auto_quantize/api.py` / `scope.py` / `model_load/`。
> **与 `main.py --analysis` 的区别**见本文 §1 与 `20-analysis.md` 文首指针。
>
> **本工具只生成精度计划 JSON，不导出量化权重。** 真正量化仍走 MoE-Quant / examples / 其它后端。

## 1. 定位：何时用 / 何时不用

| | `utils.auto_quantize` | `main.py --analysis`（`20`） |
|--|--|--|
| 目标 | 在 `--effective-bits` 预算下，自动决定搜索范围内模块 **INT4 还是 INT8** | 诊断敏感层，指导 `--ignore` / 换算法 |
| 产出 | `autoquantize_plan.json`（完整 `module_bits`）+ Top-10 敏感度报告 | 文本层排序报告 |
| 是否量化 | 否（假量化评分） | 否；部分模式要**已有**量化产物 |
| 典型下游 | 读 plan → 制定混合 W4/W8 方案 → 再跑 GPTQ/RTN/examples | 改 ignore 后重跑 PTQ |

**选用 AutoQuantize 当**：要做 **混合 W4A8（或 W4/W8 分配）**、希望机器给出按组位宽，而不是手猜 ignore。  
**选用 `--analysis` 当**：量化已跑完、掉点排查、要找该忽略哪几层。

---

## 2. 使用限制（硬约束）

1. **环境**：需适配当前 NVIDIA/DCU 的 PyTorch，并安装 `nvidia-modelopt[hf]==0.47.0`（版本以 README 为准；保留厂商 PyTorch）。
2. **工作目录**：从 **one-click-quant 仓库根**运行；`--output` **不得覆盖已有文件**。
3. **模型输入**：本地 HF 目录（`config.json` + safetensors）。支持浮点；已接入解码的 Block FP8 / MXFP4 等可分析（须先解码到 FP16/BF16/FP32 工作精度）。架构须被 Transformers + ModelOpt 支持。
4. **单卡模式**：默认 `--parallel single` 需单卡装下**完整解码模型 + 评分开销**。装不下则不得强制坚持单卡。
5. **EP 模式（`--parallel ep`）**：
   - **仅**已实现 runtime：`mimo_v2`、`kimi_k3`、`kimi_k31`（见 `model_load/__init__.py`）
   - 须 `torchrun` 多进程；**专家数 % nproc == 0**；校准序列块数须为进程数的正整数倍
   - **无通用 TP**；未注册 EP 的模型不要传 `--parallel ep`
6. **候选配方固定**：权重 channel INT4/INT8 + 动态 per-token A8（见 plan `policy`）；不是任意 scheme 搜索器。
7. **校准**：文本决定梯度敏感度与 MoE 路由覆盖；`samples`/`seq_length` 影响分数与覆盖，**绝对分数不可跨设置横比**。
8. **输出语义**：score 是梯度扰动**代理**，不是实测 ΔPPL；Top-10 报告**不能**代替主 plan 做量化。
9. **每次跑完 AutoQuantize 必须**用 `skills/scripts/plan_to_scheme.py` 生成同目录旁的 `autoquantize_scheme.json`；后续量化以该 scheme 报告为准（见 §6），不要只拿 Top-10 敏感度报告开跑。

---

## 3. 使用方法

### 3.1 依赖

```bash
python -m pip install "nvidia-modelopt[hf]==0.47.0"   # 勿覆盖厂商 PyTorch
```

### 3.2 单设备（通用 / 能装下的模型）

```bash
python -m utils.auto_quantize \
  --model /path/to/model \
  --calibration-data open-platypus \
  --calibration-samples 8 \
  --calibration-seq-length 512 \
  --effective-bits 4.2 \
  --device cuda:0 \
  --output /path/to/analysis/autoquantize_plan.json
```

- `open-platypus` 可换成本地 `.txt` / `.json` / `.jsonl`（每行一条文本字段）。
- 需要自定义代码时加 `--trust-remote-code`。
- DCU 设备名仍用 `cuda:0`（配合 `HIP_VISIBLE_DEVICES`）。

### 3.3 EP（仅 MiMo / Kimi）

```bash
CUDA_VISIBLE_DEVICES=0,1 HIP_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m utils.auto_quantize \
  --parallel ep --trust-remote-code \
  --model /path/to/mimo_or_kimi \
  --calibration-data open-platypus \
  --calibration-samples 8 --calibration-seq-length 512 \
  --effective-bits 4.2 --device cuda:0 \
  --output /path/to/autoquantize_plan.json
```

- 可加 `--dry-run`：只做元数据/分片预检，不解码大权重、不写计划。
- 可选：`--mxfp4-decode`、`--prefetch-blocks` 等（**仅 EP**；见 `--help`）。

### 3.4 常用参数

| 参数 | 含义 |
|------|------|
| `--effective-bits` | 搜索权重平均位宽上限，范围 **4～8**（预算，不是带宽） |
| `--exclude` / `--include` | 额外保留浮点 / 缩小搜索范围（源 config ignore 仍生效；exclude 优先） |
| `--policy` | JSON：仅允许 `exclude` / `include` / `grouping_rules` |
| `--calibration-samples` | 最多生成的 token **序列块**数（非原始行数） |
| `--dtype` | 分析工作精度：`bfloat16`（默认）/ `float16` / `float32` |

完整列表：`python -m utils.auto_quantize --help` 与工具 README。

### 3.5 一次运行写出的文件

| 文件 | 用途 |
|------|------|
| `--output`（主 plan） | **完整** `module_bits` / `groups` / `policy` / 覆盖与成本；量化决策以此为准 |
| 同 stem 的 `.sensitivity.txt` | 人读 Top-10 组（按 W4A8 分数） |
| 同 stem 的 `.sensitivity.json` | Top-10 结构化；含 `field_descriptions` |

---

## 4. 新模型适配规则

按需分层，**不要**混用 preprocess / MoE-Quant 注册表。

### 4.1 通用单卡路径（多数 Dense / 标准 MoE）

若 Transformers 能加载为普通 `nn.Linear`（或已支持的 3D fused experts），且单卡装得下：

1. 直接跑 §3.2；缺分组时用默认 `ModelAdapter` + ModelOpt 内置 grouping。
2. 若报 `Unsupported matrix module ...`：为该模块写 `ModelAdapter.targets_for_module`，或 `--exclude` 掉。
3. 融合组（q/k/v、gate/up、同层 experts）须同 bit：用 `grouping_rules`（capture group 正则）或子类 `grouping_rules`。
4. `in_features % 8 != 0` 的目标默认只允许 INT8（`allowed_bits`）。

### 4.2 需要自定义加载 / EP

| 扩展点 | 位置 | 何时 |
|--------|------|------|
| 源权重解码 / 架构加载 | `utils/auto_quantize/model_load/<family>.py` + `get_model_loader` | Block FP8、MXFP4、非标准 checkpoint |
| EP runtime | 同文件 `*Runtime` + `_MODELS` 注册 | 装不下单卡、需专家分片逐块加载 |
| 范围 / 分组 / 排除 | 子类 `ModelAdapter`（如 `MiMoScopeAdapter` / `KimiScopeAdapter`） | 特殊 fused 名、整层 experts 一组决策 |

注册 EP：在 `model_load/__init__.py` 的 `_MODELS` 增加 `model_type → (模块, Runtime类)`。  
**未注册则禁止对该模型宣称「可 EP」。**

### 4.3 适配检查清单

- [ ] `config.json` 的 `model_type` / architectures 已确认
- [ ] 单卡试通或已挂 EP runtime
- [ ] 源 ignore / `--exclude` 覆盖 vision、mtp、router 等不应搜的层
- [ ] `--dry-run`（EP）或小 `--include` smoke 通过后再全量
- [ ] plan `coverage` / `uncertain_groups` 可接受后再制定方案

---

## 5. 结果分析

### 5.1 先读主 plan，再读 Top-10

1. **`coverage`**：`complete=true` ≠ 所有专家都被校准触发；看 `uncertain_groups` / `unobserved`
2. **`effective_bits_target` vs `actual_effective_bits`**：实际平均位宽；未用满预算属离散组选择正常现象
3. **`module_bits`**：每个搜索目标的最终 4 或 8（量化执行的权威字段）
4. **`groups[*].scores`**：组级 W4/W8 代理分；`upgrade_gain ≈ S4−S8`
5. **敏感度 TXT**：仅 Top-10，按 **W4A8 分数**排序——**不是**升级收益排序，也不是最终升 W8 的充分条件

### 5.2 表格列怎么读（敏感度报告）

| 列 | 含义 |
|----|------|
| Rank | 按 W4 分数降序 |
| Group | **一次统一位宽**的决策组（可能是整层全部 routed experts） |
| W4A8 / W8A8 score | 敏感度代理；越大越敏感；非实测 loss |
| S4−S8 | 升 W8 的代理收益 |
| Bit | 预算求解后的选定位宽 |
| Params / Members | 计入预算的权重元素数 / 组内模块数 |
| Coverage | `unobserved=N` 等；`zero_scores` ≠ 可安全 INT4 |

### 5.3 预算直觉（候选仅 4/8 时）

```text
B_actual = 4 + 4 * p8    →   p8 ≤ (B_target − 4) / 4
```

例：`4.2` → 至多约 5% 权重元素为 W8。组大小不等时必须用加权，不能「数层数」。

---

## 6. 从 plan 到 scheme（必做）再量化

AutoQuantize **只**写 `autoquantize_plan.json`。每次跑完后**必须**再生成 `autoquantize_scheme.json`，后续量化按 scheme（审过 `_exceptions`）执行，而不是直接啃完整 `module_bits` 或 Top-10 报告。

```text
源模型 → auto_quantize → autoquantize_plan.json
       → plan_to_scheme.py → autoquantize_scheme.json   ← 必做
       → 按 scheme 角色多数位宽 + _exceptions 翻译到后端
       → GPTQ/RTN/打包 → check_artifact + PPL
```

### 6.1 转换命令

单文件脚本（内置通用 MoE fnmatch + 位宽 atom 标签；**本脚本** Python ≥ 3.8 即可；量化执行仍须仓库 Python ≥ 3.10，见 `01`）：

```bash
python3 ./skills/scripts/plan_to_scheme.py \
  --plan /path/to/autoquantize_plan.json \
  --out  /path/to/autoquantize_scheme.json
```

| 产物 / 选项 | 角色 |
|-------------|------|
| `skills/scripts/plan_to_scheme.py` | plan → 角色级 scheme 报告（默认零外部依赖） |
| `--out` 写出的 `autoquantize_scheme.json` | **后续量化的权威方案报告** |
| `--profile`（可选） | 覆盖内置 role→fnmatch；JSON 形如 `{"patterns":{role:[...]}}` |
| `--atoms`（可选） | 覆盖内置 atom id 集合；JSON 形如 `{"wo":{...},"wa":{...}}` |

规则摘要：

- 按内置（或 `--profile`）fnmatch 把 `module_bits` 映到 role
- 每 role 取 **多数** bit → 映射到 weight atom 标签；A8 来自 plan `policy.activation_bits`
- **role 级方案表达不了逐层混位**：少数派层进 `_exceptions`，开跑前必须人工审

命名非常规时：用 `--profile` 扩 pattern 后再转换；禁止未经适配强制套用未匹配结果。

### 6.2 落到本仓库执行链

1. 确认 plan `model` 指纹与当前源模型一致；`coverage` / `uncertain_groups` 可接受  
2. 跑 §6.1，打开 `autoquantize_scheme.json`：读 `roles`（多数票）+ `_exceptions` + `_unmapped_module_bits`  
3. 翻译到后端（scheme 是报告，不是 `main.py --scheme` 现成键）：
   - **仅路由专家 W4A8**：多数 INT4 落在 experts、其余应保持浮点 → `--quantize_only_experts` + pack A8（见 `13` §3.1）
   - **混合 W4A8**：INT4 + INT8 并存 → `--force_int8` + pack A8
   - **examples 离线脚本**：按该目录 README / 单脚本参数写入 `mixed_precision`（或等价分组），一次跑完
4. 跑量化 → `check_artifact` → 按 `04` 做 PPL；**ΔNLL < 0.1** 才算精度正常，否则按 `00` F / `05` 升级  
5. 例外层过多或 unmapped 偏大：加大校准 / 调 `effective-bits` 重跑 AutoQuantize，或手工覆盖例外层后再量化

---

## 7. 与其它文档的分工

| 文档 | 职责 |
|------|------|
| **本文 `22`** | AutoQuantize 限制、用法、适配、读结果、制定方案 |
| `20` | `main.py --analysis` 五模式诊断 |
| `13` | MoE-Quant 执行与仅路由专家 · 混合 W4A8 |
| `05` | 精度掉点排查阶梯（量化后） |
| 工具 README | 参数细节、预算公式、校准数据说明 |

## 8. 红线（本工具）

- 不得把 Top-10 敏感度报告当成完整量化配方  
- 不得对未注册 EP 的模型使用 `--parallel ep`  
- 不得伪造 plan 或跳过 `coverage` 审查直接开向量化  
- 不得声称 AutoQuantize「已完成量化」——量化交付须产物契约 + `04` §3.2（除非用户明确跳过评测；见 `SKILL`）
