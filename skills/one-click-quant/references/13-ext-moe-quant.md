# 扩展：MoE-Quant（`utils/MoE-Quant/`）

> **权威**：`utils/MoE-Quant/README.md`、`src/models/base.py`、`src/models/registry.py`、
> `quant.py` / `pack_quantized_model.py`、`tests/run_*.sh`。
> 本文：**两阶段流程、范围/位宽术语、adapter 契约、新增模型**。模型族特例以 adapter 报错为准。

## 1. 定位

有校准的超大 MoE GPTQ（常见 INT4 / 混合 INT4+INT8）。工作目录必须是 `utils/MoE-Quant/`
（或 `PYTHONPATH=utils/MoE-Quant`）。支持列表 **仅以** `registry.py` 的 `_ADAPTER_CLASSES` 为准：

```bash
grep -n "_ADAPTER_CLASSES" -A30 utils/MoE-Quant/src/models/registry.py
ls utils/MoE-Quant/tests/run_*w4a8*.sh 2>/dev/null
```

## 2. 两阶段

```bash
cd utils/MoE-Quant

# ① GPTQ → <save_dir>/<layer>/quantized_weight.pt + metadata.pt
# NPROC / 校准集 / 是否 --quantize_only_experts|--force_int8|--ignore|--group_size
# 以该模型 adapter 与 tests/run_*.sh 为准，勿照抄骨架默认值
torchrun --nnodes=1 --nproc-per-node="$NPROC" --master_port "$MASTER_PORT" quant.py \
  --model_name_or_path "$MODEL" --dataset_name_or_path "$CALIB_SET" \
  --num_calibration_samples 512 --max_sequence_length 4096 \
  --bits 4 --rel_damp 0.1 --sym --dtype bfloat16 --save_dir "$QUANT"

# ② 打包；A16 → W4A16；A8 → 仅路由专家 / 混合 W4A8 或 MoE-Quant W8A8（§3）
python3 pack_quantized_model.py \
  --model_name_or_path "$MODEL" --quantized_model_path "$QUANT" \
  --packed_model_path "$QUANT-packed" --dtype bfloat16 --activation-bits 16
```

- ① 产物不是可加载 HF；必须跑 ②
- ② 的 `--model_name_or_path` 是**原始源模型**
- `metadata.pt` 中的 `bits` / `group_size` / `quantize_only_experts` / `ignore` / `force_int8` 会**覆盖**② 同名 CLI，且强制 `sym=True`
- 源为 BF16 时①须显式 `--dtype bfloat16`（① 默认 `float16`）
- 优先跑已有 `tests/run_*.sh`，少手拼
- 若需**自动**决定哪些层 INT4 / INT8：先跑 AutoQuantize 出 plan，再映射为 `--force_int8` / `--quantize_only_experts`（见 `22`）

## 3. 范围与位宽（术语唯一出处）

### 3.1 仅路由专家 W4A8 · 混合 W4A8

本 skill **只保留这两档 W4A8 名称**；不再使用「纯/裸 W4A8」。

| 名称 | 含义 | 典型做法（MoE-Quant） |
|------|------|------------------------|
| **仅路由专家 W4A8** | 只量化**路由 experts**（INT4）；其余 Linear **浮点**；动态 A8 | `--bits 4` + `--quantize_only_experts` → pack `--activation-bits 8` |
| **混合 W4A8** | experts INT4 + 其它参与量化的 Linear **INT8** + 动态 A8 | `--bits 4` + `--force_int8 ...`（或 adapter 固定配方）→ pack `--activation-bits 8` |

- **W4A16**：权重 INT4、激活非 A8；MoE 上常配 `--quantize_only_experts`，pack `--activation-bits 16`
- 两档均须 **registry / 官方脚本支持** 才算可行
- llmc 的 `--scheme W4A8` 是通用配方名，**不是**本 skill 的「仅路由专家 / 混合」档位；推荐 W4A8 时只推上表两档

### 3.2 开关

| 开关 | 作用 |
|------|------|
| `--quantize_only_experts` | **仅路由专家** W4A8（及常见仅专家 W4A16）的范围 |
| `--ignore` | 追加排除；`re:` 为 `re.match` 前缀匹配 |
| `--force_int8` | 匹配层 GPTQ 为 INT8，其余未 ignore 层用 `--bits`（**混合** W4A8） |
| 优先级 | `--ignore` > `--force_int8` |

- **仅路由专家**：用 `--quantize_only_experts`（或等价 ignore），不要再叠 `--force_int8` 去量化 attn/MLP
- **混合**：用 `--ignore` + `--force_int8`（或官方脚本），不要叠 `--quantize_only_experts`
- 融合组（如 q/k/v 或 gate/up）须同进同退
- 部分 adapter **禁止**用户改 ignore / force_int8 / quantize_only_experts（以 `validate_*` 报错为准）

| 目标 | 做法 |
|------|------|
| 仅路由专家 W4A8 | `--bits 4` + `--quantize_only_experts` + pack A8 |
| 混合 W4A8 | `--bits 4` + `--force_int8`（或固定配方）+ pack A8 |
| MoE-Quant W8A8（`--bits 8`） | `--bits 8` + pack A8（**不是**已废弃的「纯/裸 W4A8」） |
| 仅专家 W4A16 | `--bits 4` + 常 `--quantize_only_experts` + pack A16 |

### 3.3 模型族快照（可能过期）

> **快照**，接新模型或改 adapter 后以 `validate_quantization_args` / `validate_packing_args` 与 `tests/run_*.sh` 为准，**不要按族名类推**。

查阅方式：读对应 `src/models/*_adapter.py` 的 `validate_*`；需要 W4A8 时先 `ls tests/run_*w4a8*.sh`。

常见约束类型（举例，非完整名单）：仅允许 `bits=4`；强制 experts-only；禁止用户改 ignore；强制 `activation-bits 8`；禁 `--resume` / 空 `save_dir` 等——**以当次 adapter 为准**。

## 4. `quantization_config`

由 `pack_quantized_model.py::prepare_quantization_config` 写出：`quant_method: compressed-tensors`；
单一或混合 `config_groups`；`group_size` 省略时常落盘为 `-1`。
`quantize_only_experts`（仅路由专家档）时默认 ignore 使非 expert 保持浮点（`base.py` 一类实现）——属预期。

## 5. Adapter 契约

差异收在 `src/models/<model>_adapter.py`。常用钩子：`matches`、`checkpoint_keys_for_model_key`、
`materialize_state_dict`、`resolve_ignore` / `resolve_layer_bits`、`validate_*`、`pack_checkpoint`（可选）等。
详见 `base.py`。

## 6. 新增模型

1. 写 adapter，实现 `matches` 与 key 映射
2. 注册 `_ADAPTER_CLASSES`（**顺序敏感**：更具体在前）
3. 在 `validate_*` 写清位宽/范围；支持 W4A8 则提供 `tests/run_<model>_*.sh`
4. **不要改** `quant.py` 主流程；勿与 `preprocess/adapters/` 混用

## 7. 验收

- `quant_method: compressed-tensors`
- 仅路由专家：experts INT4 packed + 非 expert bf16 + 动态 A8
- 混合：INT4 packed 与 INT8 并存（常 `group_int4`+`group_int8`）+ 动态 A8
- `python3 ./skills/scripts/check_artifact.py <产物目录> --expect-quant-method compressed-tensors`
- 精度对比见 `04` §3.2；**ΔNLL < 0.1** 才算精度正常
