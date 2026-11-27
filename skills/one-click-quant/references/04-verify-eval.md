# 阶段4 产物校验与评测

> **权威来源**：仓库根 `README.md`「PPL评估」、`utils/lm_eval.py`、`utils/torch_infer.py`。
> 顺序：**先产物契约 → 再张量 dtype → 再 PPL（ΔNLL）**。前一步不过不要往下走。
> 精度是否合格只看 **ΔNLL < 0.1**（§3.2）。定位掉点用 `--analysis`（`20` / `05`），不把 `examples/*-MSE.py` 当精度门禁或定位主手段。

## 1. 产物契约（目录级）

产物契约按下面四条逐项看。**只有前两条是硬门禁**，后两条按方案而定：

- `config.json`：含量化配置块，键名与内容见下节
- **权重文件二选一**，不要只认 index：
  - 多分片：`model.safetensors.index.json` + `weight_map` 里列出的**每个分片都真实存在**
  - 单分片（小模型常见，<10GB 规模）：**只有 `model.safetensors`，没有 index.json**
- tokenizer 相关文件（`tokenizer.json` / `tokenizer_config.json` / `special_tokens_map.json` 等）：
  **随源模型一起复制**，源头没有则产物也没有 —— 这种缺失不算量化失败，但下游加载会受影响，需要单独补
- llmcompressor 路径还会写 `recipe.yaml`（量化配方快照）；自定义脚本可能写 `QUANTIZATION_DONE.txt`

### 1.1 配置键与 `quant_method` 必须对得上

| 产出路径 | 配置键 | 需要核对 |
|---|---|---|
| llmcompressor（`FP8_DYNAMIC` / `W8A8` / `W4A16` / `--scheme W4A8` 等） | 通常是 `quantization_config` | compressed-tensors；`--scheme W4A8` 是 llmc 配方名，**不是** `13` 的「仅路由专家/混合」档位 |
| MoE-Quant 打包产物 | 通常是 `quantization_config` | `quant_method: compressed-tensors`；混合 W4A8 常见 `group_int4` + `group_int8`；仅路由专家则非 expert 仍为 bf16 |
| `examples/` 下手写转换脚本 | **多为 `quantization_config`**；部分脚本（如 kimi mxfp4、部分 qwen 混合精度脚本）用 `compression_config` | 各脚本自定，**以脚本实现为准**，不要按本条推断 |
| 含 `mixed_precision` 的混合精度产物 | 常为 `compression_config`（可能同时复制进 `text_config`） | `mixed_precision.{experts,w8a8_include}` 等；若双写则两处应一致；`quant_method` **以脚本写出为准** |

只看"有没有这个键"不够：**键对了但 `quant_method` 不对，说明走了另一条路线**。以本表 + 当次方案预期核对。

> **配置键名本身不是判据**：`quantization_config` 与 `compression_config` 在流通的产物里都能见到
> （不同 alg / 不同 `examples/` 脚本写不同的键）。校验脚本因此两个键都接受，但必须打印并核对 `quant_method`。

> **不要因为缺少 `model.safetensors.index.json` 就判定产物失败**：权重能装进单个分片时
> `save_pretrained` 只写 `model.safetensors`，不生成索引。按 index 判断会把正常产物误判成坏的。

`QUANTIZATION_DONE.txt` **不是 PTQ 判据**：它只由 `examples/` 下少数模型专属转换脚本写出（内含 stats JSON），
`main.py` 的任何 PTQ 主链路都不产出该文件。哪些脚本会写，以 `grep -rl QUANTIZATION_DONE --include=*.py .` 的实际结果为准，
不要记忆这份名单。

手工校验（把产物目录作为参数传进去，兼容单分片与多分片）：

```bash
python3 - "<产物目录>" <<'PY'
import glob, json, os, sys

d = sys.argv[1]
cfg = json.load(open(os.path.join(d, "config.json")))
has_q = "quantization_config" in cfg or "compression_config" in cfg
qkey = "quantization_config" if "quantization_config" in cfg else "compression_config"

idx_path = os.path.join(d, "model.safetensors.index.json")
if os.path.exists(idx_path):
    wm = json.load(open(idx_path))["weight_map"]
    missing = sorted({s for s in wm.values() if not os.path.exists(os.path.join(d, s))})
    print("layout: sharded  tensors=%d shards=%d missing=%s"
          % (len(wm), len(set(wm.values())), missing))
else:
    single = glob.glob(os.path.join(d, "model*.safetensors"))
    files = [os.path.basename(x) for x in single]
    missing = [] if single else ["NO WEIGHT FILE"]
    print("layout: single   files=%s missing=%s" % (files, missing))

print("quant config:", qkey if has_q else "MISSING",
      "| quant_method =", cfg.get(qkey, {}).get("quant_method") if has_q else None,
      "| format =", cfg.get(qkey, {}).get("format") if has_q else None)
print("tokenizer:", os.path.exists(os.path.join(d, "tokenizer.json")))
print("recipe.yaml:", os.path.exists(os.path.join(d, "recipe.yaml")))
PY
```

输出示例（单分片产物）：

```text
layout: single   files=['model.safetensors'] missing=[]
quant config: quantization_config | quant_method = <本方案应有的值> | format = None
tokenizer: True
recipe.yaml: True
```

判读：`missing` 必须为空；`quant_method` 必须与方案对应（对照 1.1 的表）。
部分路径即使只有一个分片也会写 `model.safetensors.index.json`，此时会打印 `layout: sharded ... shards=1`，属正常。

**优先直接跑校验脚本**（等价于上面这段手工检查，另加分片存在性与 dtype 抽样）：

```bash
python3 ./skills/scripts/check_artifact.py <产物目录> --expect-quant-method <期望值>
```

上面的 heredoc 只在脚本不可用时用；`--expect-quant-method` 省略则只报告实际值不比对。

> **适用范围（L5/C2）**：`check_artifact.py` 只用于**量化产物**
> （`main.py` PTQ / MoE-Quant 打包 / `examples/` 量化脚本的最终档）。
> 下列情况**不要**跑它当门禁：
> - `--alg preprocess` 浮点目录（无量化配置块 → 必然 FAIL；判据见 `10-ext-preprocess.md`）
> - 只改了 `config.json`、权重未变的中间手术步（先按该脚本 README 约定验）
> - 尚未写完分片的半成品目录

## 2. 张量 dtype（结构级）

```bash
python3 tools/model_xray.py --input_dir <产物目录> --output_dir <xray输出目录>
```

逐条核对**方案预期**，而不是"看起来像就行"。不同方案的落盘形态差别很大，**别用一套键名去套所有方案**：

| 方案 | 权重键名 | 权重 dtype | scale 键名 | scale dtype |
|---|---|---|---|---|
| `FP8_DYNAMIC` | `...weight` | `float8_e4m3fn` | `...weight_scale` | bf16 `[N,1]` |
| `W8A8`（per-channel INT8） | `...weight` | `int8` | `...weight_scale` | bf16 |
| `MoE-Quant` / WNA16（pack-quantized） | `...weight_packed` | **int32**（8×int4 装进 1 个 int32） | `...weight_scale` | bf16 |
| `MoE-Quant` 混合 W4A8（`group_int4`+`group_int8`） | INT4 层 `weight_packed`；INT8 层 `weight` | int32 / int8 | 配对 `weight_scale` | 多为 bf16 |
| **`mixed_precision` 混合精度** | 同一产物里**多套并存**（按分组） | 逐类核对 | 按类配对 | 按类 |

> 若配置是 `compression_config.mixed_precision.{experts, w8a8_include}` 等分组结构，
> **不能**用「全模型一个 dtype」的眼光看；分组语义与层范围以写出该产物的脚本 / README 为准。

落盘形态示例（**shape 仅示意**，以本产物 xray 为准）：

```text
# FP8_DYNAMIC / W8A8：...weight 为 fp8/int8 + ...weight_scale；ignore 层仍为 bf16
# MoE-Quant pack-quantized：...weight_packed (int32) + ...weight_scale；未量化层 bf16
```

判据：

- 该量化的层：权重 dtype 是 int8 / fp8 / int32-packed，且**必有配对 scale**
- 不该量化的层（`modules_to_not_convert` / `ignore` / `targets` 之外，以及所有 `*norm`、`embed_tokens`、`lm_head`）：仍是 bf16
- 不存在残留 `weight_scale_inv`（block FP8 源格式痕迹，说明解包没做干净）
- MoE 的 K 维是否减半：int8 packed → K/2；int32 packed → K/8；纯 INT8 → K 不变。**用 `weight_shape` 或已知 `intermediate_size` 交叉验证**
- fused 3D int8 + scale 若出现在产物里，按三条不变式验收即可，并以该路径文档/脚本为准，不要凭键名形态猜路线

判据不一致 → 回阶段3 查 ignore 列表与 `targets`，不要在错的产物上做评测。

## 3. PPL / 下游评测

> 以下为**本仓库** `utils/lm_eval.py` / `main.py --lm-eval` 封装限制，不是通用评测规范。

- 强制 `HF_DATASETS_OFFLINE=1`
- 缓存须已在 `HF_DATASETS_CACHE`（默认 `./datasets`）下；缺则 `sys.exit(1)`（目录名以脚本为准）
- backend / 任务 / batch **写死**（当前为 vllm + wikitext + batch 1）

因此：该封装依赖不可用时，**不要**反复重试 `main.py --lm-eval`；可改用 §3.1 的 `torch_infer`，
或绕过封装直接调 `lm_eval`（本仓库不保证后一条）。

### 3.1 回退路径：`utils/torch_infer.py`

vLLM 不可用 / 只想看生成是否退化时，用纯 PyTorch 反量化推理：

```bash
python3 -m utils.torch_infer --model <产物目录>
python3 -m utils.torch_infer --model <产物目录> --prompt "你好,1+1等于几" --max-tokens 64
python3 -m utils.torch_infer --model <产物目录> --no-dispatch
```

- `--prompt` 逗号分隔多个 prompt；不传则跑内置中英测试集
- `--no-dispatch`：`compressed-tensors` 的 `dispatch_model` 报错时使用
- `--dtype` / `--device` 可覆盖（CPU 可跑，但慢）

已知问题（遇到时**不得直接判定产物损坏**，须先核对下列情形）：

- 部分 W8A16 / `compressed-tensors` 产物的 `config_groups.*.input_activations` 为 `null` 时，
  `_print_model_info` 会 `AttributeError: 'NoneType' object has no attribute 'get'`
  （`utils/torch_infer.py`）。这是工具打印配置时的空指针，不是权重损坏；可换 GPU/`--lm-eval`，
  或等上游修（`a = gcfg.get(...) or {}`）
- 在**非目标加速器**或纯 CPU 上回退路径偶发乱码输出：**乱码 ≠ 量化失败**。
  先用 `check_artifact` + `model_xray` 核对契约与 dtype，再在目标加速器上复跑或改看 PPL

回退路径只证明"能加载、能生成、无明显退化"，**不能替代 PPL 数值结论**，报告里要写清楚用的是哪条路径。

### 3.2 精度对比报告（ΔPPL / ΔNLL）与合格门槛

做完产物契约（§1）与 dtype（§2）之后，按下面流程出**对比报告**。

| 符号 | 定义 |
|------|------|
| PPL | lm_eval wikitext 表格里的 `word_perplexity` |
| `PPL_b` / `PPL_q` | baseline / 量化产物的 word_perplexity |
| ΔPPL | `PPL_q − PPL_b` |
| ΔNLL | `ln(PPL_q) − ln(PPL_b)`（nats） |

**精度正常（本 skill 口径）**：同结构 baseline 对比下 **ΔNLL < 0.1**。  
**量化交付**默认必须跑完本节（`SKILL` 硬门禁）；仅用户明确跳过评测时可省略。  
契约 / dtype 仍是前置硬门禁；**ΔNLL ≥ 0.1 不得宣称精度合格**，须按 `00` F1/F2 或 `05` 继续升级。

**完成精度验证的最低交付**：四个数都要有 —— `PPL_b`、`PPL_q`、`ΔPPL`、`ΔNLL`，并给出是否 `ΔNLL < 0.1`。只报单侧 PPL 不算完成。

Agent 流程：

1. `python3 ./skills/scripts/check_artifact.py <产物目录>` 退出码 0
2. 确认 `HF_DATASETS_CACHE`（默认 `./datasets`）下存在 `EleutherAI___wikitext_document_level`
3. 对 **baseline** 与 **产物** 各跑一次 `--lm-eval`（同一 `tasks=wikitext`、同一离线缓存；
   `tensor_parallel_size` / `dtype` / `gpu_memory_utilization` 尽量一致），从输出表摘 `word_perplexity`
4. 用脚本算差并打印（脚本不跑评测；会标注 `precision_ok`）：

```bash
# baseline / 产物各跑一次；pretrained= 换成真实目录
python3 main.py --lm-eval "pretrained=<baseline>,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.6"
python3 main.py --lm-eval "pretrained=<产物>,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.6"

python3 ./skills/scripts/report_ppl_delta.py \
  --baseline-ppl <PPL_b> --quant-ppl <PPL_q>
```

报告模板（可原样贴给客户）：

```text
PPL_b   = <baseline word_perplexity>
PPL_q   = <quant word_perplexity>
ΔPPL    = PPL_q - PPL_b
ΔNLL    = ln(PPL_q) - ln(PPL_b)   # nats
precision_ok = yes|no            # yes 当且仅当 ΔNLL < 0.1
```

客户问「是否达标」：以 **ΔNLL < 0.1** 回答；不达标则进 `05` / `00` F 升级，不要改口无门槛。

## 4. 评测对比基线的纪律

- 对比必须是**同结构同层数**的 BF16 源（或同结构预处理浮点），不能拿全量模型 PPL 去比 mini 产物
- 精度结论必须给出 §3.2 的四个数 + 是否 ΔNLL < 0.1；缺任一等于没结论
- ΔNLL ≥ 0.1：走 `05-precision-debug.md` / `00` Dense·MoE 升级阶梯，不要直接改方案或谎报合格
