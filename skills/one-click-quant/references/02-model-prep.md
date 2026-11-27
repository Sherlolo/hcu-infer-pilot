# 阶段2 模型准备

> **权威来源**：各 `examples/*/build_mini_*.py` 与 `tools/model_xray.py` 的 `--help`；参数以当次 `--help` 为准。
> 本文只写判断顺序、关键 flag 与门禁。
>
> **尚未选定量化路径时**：先走 [00-new-model-playbook.md](00-new-model-playbook.md)
> （判适配=量化历史 → 菜单或 Dense/MoE/超大链路 / 自提评估，见 `00`），客户确认后再做本文的解包 / mini。

## 何时用本文

- 量化方案已定（或客户已确认 `00` 的推荐），需要解包 / 建 mini / 核对结构覆盖
- 不确定权重是 BF16 / block FP8 / MXFP4 / 已量化（也可作为 `00` 取证的补充手段）
- 需要一个小模型来验证量化链路（而不是动辄几 T 的全量权重）
- 复用别的模型的 mini 做冒烟，但担心结构覆盖不全

## 步骤 1：权重取证

```bash
python3 tools/model_xray.py --input_dir <模型目录> --output_dir <输出目录>
```

产出每个分片的 `<shard>.metadata.json`，记录每个 tensor 的 dtype 与 shape。

三个必须知道的行为（都影响怎么用、怎么判）：

- 它逐个 `get_tensor` 读出 dtype/shape，**不是只读 header** —— 不会常驻整个模型，但会加载**当前分片**的全部张量
- `--output_dir` 省略时，JSON **写回模型目录本身**。而 `model_free_ptq` 等走 `cached_file` 的路径
  要求模型目录里没有非权重杂项（见 `03-quant-runbook.md` §3），所以**必须显式给 `--output_dir`**，
  否则下次量化会报 `OSError: ... does not appear to have a file named <shard>.metadata.json`
- 它**不返回非零退出码**（出错只打印 `❌`）。判据是生成的 JSON 内容，不是 exit code

判据：

| 观察到 | 结论 | 下一步 |
|---|---|---|
| 有 `weight_scale_inv` | block FP8（如 DeepSeek 官方） | 先解包到 BF16 |
| 有 `weight_packed` + `weight_shape` + `weight_scale`，`weight_packed` 是 **int32**，`weight_scale` 是 **bf16** | compressed-tensors WNA16 pack-quantized（MoE-Quant / W4A16） | 解量化或 `--keep-quant` |
| 有 `weight_packed` + `weight_scale`，且 `weight_scale` 是 **uint8** | MXFP4（E8M0 指数，group_size=32） | 解包；Kimi 可用 `build_mini_kimi_k3.py --src-format mxfp4` |
| 有 `...experts.gate_up_proj` / `...experts.down_proj` 3D 键，dtype **int8** + 同名 `weight_scale` float32 | 已量化的 fused 形态 | 已是成品，不得当作源模型再次量化 |
| 有 `quantization_config` / `compression_config`，权重是 int8 / fp8 | 已量化（llmcompressor / MoE-Quant / examples 等） | 同上 |
| dtype 全 bf16/fp16，无 scale | 浮点权重 | 可直接量化 |
| 有 3D 键 `...mlp.experts.gate_up_proj` / `...down_proj`，dtype bf16、无 scale | 浮点权重，且 experts 是 **fused** 形态 | 可直接量化（下游路径是否拆 fused 以该路径文档为准） |

判据要点：

- **只看 `weight_packed` 分不出 MXFP4 和 INT4** —— 要看 `weight_scale` 的 dtype（uint8=E8M0 → MXFP4；浮点 → affine）
- **scale 的浮点 dtype 不是固定 float32**：WNA16 / MoE-Quant 产物常见 bf16；其它路径可能是 float32。
  判「是否 affine 量化」只看"是不是浮点"，别写死 float32
- **fused 3D** 常见于源模型 bf16 裸权重（`...experts.gate_up_proj` / `...experts.down_proj`），
  或其它工具产出的已量化形态。验收时按 `04-verify-eval.md` 第 2 节不变式核对
- **判"是不是已量化 / 走了哪条路线"最可靠的是看 `quant_method`**，而不是猜键名形态：
  `quantization_config` / `compression_config` 键名本身不是判据，`quant_method` 才是下游加载器真正读的字段
  （对照 `04-verify-eval.md` 1.1 的表）

## 步骤 2：解包 / 反量化分支

| 源格式 | 机制 | 落点 |
|---|---|---|
| DeepSeek block FP8 | recipe 中 `unpack_fp8`（**默认就是 `true`**，设 `false` 才关闭），多卡解包到 `{save-dir}.bf16_staging`，分片 1:1 | `preprocess/unpack/fp8.py` |
| 已量化 W4A16 | `build_mini_*` 可识别 `--src-format w4a16` | `examples/*/build_mini_*.py` |
| 已量化 MXFP4 | 一次性转换脚本，直接读 safetensors 重打包 | `examples/kimi/kimi_k3_mxfp4_to_int4.py` |

- `unpack_fp8_gpus` 可覆盖解包卡数（默认跟随 `--preprocess-gpus`）。
- 极小冒烟若 recipe 支持 `unpack_fp8_inline: true`，可在 adapter 内原地解包（是否支持以该 recipe 为准）。

> **是否有 preprocess / 某 adapter**：以 `ls preprocess/adapters/`、`ls preprocess/recipes/` 为准，
> 不以文档记忆为准。无对应文件则走 `examples/` 转换脚本或 MoE-Quant（见 `10`）。

## 步骤 3：构建减层 mini

```bash
ls examples/*/build_mini_*.py    # 脚本清单以实际为准
python3 <脚本> --help            # -e 等默认值因脚本而异
```

**脚本与目标模型族必须对上**（不能拿 A 族 mini 脚本冒烟 B 族）。下表为**快照概览**，以 `--help` 为准：

| 脚本（示例） | 目标族 | `-e` 默认（快照） |
|---|---|---|
| `examples/qwen/build_mini_qwen3_8.py` | 该脚本服务的 Qwen MoE 变体 | 见 `--help` |
| `examples/deepseek/build_mini_deepseek_v3.py` | DeepSeek-V3 类 | 见 `--help` |
| `examples/glm/build_mini_glm5_3.py` | GLM 类 | 见 `--help` |
| `examples/kimi/build_mini_kimi_k3.py` | Kimi 类 | 见 `--help` |

公共参数：

- `--src` / `--dst`：源模型目录 / 输出目录
- `--src-format`：`auto`（默认；具体可选值因脚本而异，见「差异」）
- `-n/--num-layers`：保留前 N 层，默认 4
- `--layer-range START:END`：截取源层区间（半开）
- `--layer-ids 0,12,45,92`：显式指定不连续层号（与上两项互斥）
- `-e/--num-experts`：每层保留 expert 数（冒烟时可远小于全量；具体默认见 `--help`）
- `--keep-quant`：保留源量化张量与 `quantization_config`；默认解量化为 BF16

差异（**以各脚本 `--help` 为准，不要一刀切**）：

- `--src-format` 可选值因脚本而异
- `--dry-run` / `--no-mtp` / `--include-nextn` 等并非所有脚本都有

行为要点：

- 保留层会被**重映射为连续** `model.layers.0..N-1`
- 默认解量化后常 `pop("quantization_config")`，但多数脚本**不处理 `compression_config`** ——
  源若带该键，交付前须手工删或按 `10` 验收（两个量化键都不存在）

**嵌套 `text_config`**：多数 `build_mini_*` 只读顶层字段，嵌套 config 会 `KeyError`；
若某脚本支持 `config.get("text_config", config)` 回退，以该脚本实现为准。不确定时先 `--dry-run` 或拍平字段。

```bash
python3 <build_mini脚本> --src <源> --dst <输出> -n 4 -e 8 --dry-run   # 若支持
python3 <build_mini脚本> --src <源> --dst <输出> -n 4 -e 8
```

## 结构覆盖要求（硬门禁）

> 减层**不等于**取前 N 层。必须覆盖**该模型实际拥有**、且接入路径会碰到的每一种结构单元。
> 候选清单（有则覆盖，无则跳过；不得为不存在的结构无依据地拼凑层名）：
> MLA attention / dense MLP / MoE experts / shared_experts / MTP / shared_head.norm / linear_attn。

对 **dense 小模型**（无 MoE / 无 MTP / 无 MLA）只需覆盖其真实模块（如 `self_attn` + `mlp` + norm），
对照 `config.json` / `model_xray` 键名勾选即可。对 MoE / 多模态 / MTP 模型，只取前 4 层时常漏掉
晚出现结构，冒烟会假通过——层号不确定时用 `--layer-ids`，或 `--dry-run` 先看筛选结果。

## 门禁

- [ ] `--dst` 下 `config.json` 可解析，`num_hidden_layers` 等于设定值
- [ ] `model_xray` 复查：dtype 全为 bf16/fp16，无 `weight_scale_inv`（看 JSON，不看退出码）
- [ ] 权重文件存在且自洽：多分片时 `model.safetensors.index.json` 的 `weight_map` 指向的**每个分片都真实存在**；
      单分片（小模型常见）只有 `model.safetensors`、**没有 index.json**，这不是失败（判据见 `04-verify-eval.md`）
- [ ] 该模型**实际存在**的结构单元至少各出现一次（无则跳过，有则必须覆盖）
