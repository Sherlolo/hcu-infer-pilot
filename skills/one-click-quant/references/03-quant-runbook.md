# 阶段3 量化执行手册

> **权威来源**（本文与代码冲突时，以这些为准）：
> `python3 main.py --help`、仓库根 `README.md`（含「常见故障排查」）、`utils/MoE-Quant/README.md`、
> `templates/quant_templates.yaml`。
> 本文只写**路径怎么选、命令怎么写、哪些是已知问题**。
>
> **方案未定**：先走 [00-new-model-playbook.md](00-new-model-playbook.md)（判适配 / 菜单 / 自提评估 / 未适配推荐）；
> **选具体方法**查 [06-method-model-map.md](06-method-model-map.md)（模型 ↔ scheme/alg / examples / MoE-Quant）。
> 让客户确认主方案，再回到本文写具体命令。未确认前不要擅自选定 `--alg` / `--scheme` 开跑。

## 1. 路径选择

本仓库量化路径**只有三类**（先定大类，再选具体命令）。**SlimQuant 已废弃，禁止使用或推荐。**

| 大类 | 是什么 | 典型入口 |
|------|--------|----------|
| **llmc** | `main.py` + llmcompressor / `llmc_oneshot` / `model_free_ptq` | `--scheme` / `--alg ptq` / `--alg model_free_ptq` |
| **examples** | `examples/<family>/` 离线转换脚本（可含混合精度） | 该目录 README + 脚本 |
| **MoE-Quant** | `utils/MoE-Quant/` EP GPTQ 打包 | `13` + `tests/run_*.sh` |

**首选方案与升级**：以 [`00`](00-new-model-playbook.md) 总流程为准（D 打分/菜单确认 → 开跑；失败或 ΔNLL≥0.1 才 F）。本文只写确认后的**命令怎么写**。

| 提醒 | 见 |
|------|-----|
| 未适配 >500 GB / Dense 超大禁 llmc | `00` 超大规则（本流程门禁） |
| scheme/alg 语义与画像解读 | `06` |
| MoE W4A8 两档术语 | `13` §3.1 |

通用 llmc 目标（族专属 scheme 见下表，**不是**任意模型的默认菜单；首选方案仍以打分为准）：

| 目标 | 推荐路径 |
|------|----------|
| BF16 → per-channel FP8 | llmc：`--scheme FP8_DYNAMIC` |
| BF16 → per-channel INT8 | llmc：`--scheme W8A8` |
| Block FP8 → BF16 | llmc：`--scheme FP8_TO_BF16`（解包，非量化） |
| 快速免数据 PTQ | llmc：`--alg model_free_ptq`（配合 scheme） |
| 先做浮点预处理再量化 | `--alg preprocess` → 下游加 `--from-preprocessed`（**transform 须满足 `10` 可开条件**） |
| 单机 MoE GPTQ（含混合 W4A8） | MoE-Quant（命令与术语见 `13`；本节 §4 仅骨架） |
| 模型专属 / 混合精度落地 | examples（§7；以 `ls examples/` 为准） |

**仓库已有族专属 scheme 快照**（仅当取证命中对应 arch/权重形态时列入菜单；不得作为通用推荐）：

| 条件线索 | scheme |
|----------|--------|
| DeepSeek-V3 类 block FP8 | `DeepseekV3_FP8_TO_INT8` / `DeepseekV3_FP8_TO_FP8` |
| Kimi 类 W4 源 | `Kimi_W4_TO_INT8` / `Kimi_W4_TO_FP8` |
| Qwen3.5 MoE BF16 | `Qwen35_MOE_BF16_TO_INT8` |

自定义 scheme 通过 `templates/quant_templates.yaml` 解析（架构 + 尺寸签名）。更多 scheme 可参考 compressed-tensors 上游的 `quant_scheme`。

> **`--help` 里的 scheme 清单不是全集**：`main.py` 的 `--scheme` help 只列了 8 个，
> 但 `llmc_oneshot.py` 的配方表里还有 `W4A16` / `W4A16_ASYM` / `W4A8` / `W4AFP8`（都走 `smoothquant_gptq`），
> 只有落在 `quant_templates.yaml` 里的才会显示成独立 scheme。判断"支持不支持"看配方表与 YAML，不得仅依据 --help。

## 2. 最简命令

路径由用户提供。不清楚就先问，或用 `$MODEL` 模板 —— 不要编造 `/models/...`，也不要扫盘。

```bash
MODEL="<请替换为实际 HF 目录>"

# 通用 FP8 / INT8
python3 main.py --model "$MODEL" --scheme FP8_DYNAMIC --save-dir "${MODEL}-FP8_DYNAMIC"
python3 main.py --model "$MODEL" --scheme W8A8        --save-dir "${MODEL}-W8A8"

# 强制指定算法（覆盖 scheme 推导）
# model_free_ptq 的 --model 必须是相对路径（见 §3）；下面假设已按 §3 建好软链
python3 main.py --model .models/clean-model --scheme W8A8 --alg model_free_ptq \
  --save-dir "${MODEL}-W8A8-mf"
python3 main.py --model "$MODEL" --scheme W8A8 --alg ptq

# 两段式：预处理 → 再量化（--transforms 只列满足 10 可开条件的项；勿默认全开）
python3 main.py --alg preprocess \
  --model "$MODEL" --save-dir "${MODEL}-preprocessed" \
  --recipe preprocess/recipes/<model>.yaml \
  --transforms quarot \
  --preprocess-gpus 8

python3 main.py --model .models/clean-preprocessed --scheme W8A8 --alg model_free_ptq \
  --save-dir "${MODEL}-W8A8" --from-preprocessed
```

> **`--save-dir` 可以用绝对路径**（内部走 `shutil.copyfile`，不经过 `cached_file`）。
> **`--model` 路径规则**：
> - **必须相对路径**：`model_free_ptq`（见 §3）
> - **其它 alg**：默认可绝对路径；一旦出现 `does not appear to have a file named`，
>   优先按 §3 建相对软链再跑（同一类 `cached_file` 问题），不要先怀疑权重损坏

> `--alg preprocess` **必须显式给 `--recipe`**：省略时会落到一个仓库里并不存在的默认 recipe，
> 直接 `FileNotFoundError`。先 `ls preprocess/recipes/` 看有哪些可用。见 `10-ext-preprocess.md`。

不指定 `--save-dir` 时默认落在 `"{--model}-{--scheme}"`，见第 5 节。
`--from-preprocessed` 会做浮点权重自检，失败即退出 —— 不要把量化产物再喂给它。

## 3. 相对路径与干净软链（`model_free_ptq`）

`model_free_ptq` 走 `get_checkpoint_files` → `transformers.utils.cached_file`，因此：

1. **必须在仓库根目录运行**（`main.py` / 模板用相对路径加载）。
2. **`--model` 必须是相对路径，不能是绝对路径**。绝对路径经 `cached_file` 时常退化成
   `OSError: /abs/path does not appear to have a file named <basename>`。
3. **模型目录里不能有断链软链接或非权重杂项文件**（`os.walk` 会一并交给 `cached_file`）。

在**仓库根**建只含必要文件的软链目录；**必须在仓库根展开 `$SRC` 的 glob**：

```bash
SRC="<源模型绝对路径>"
cd <仓库根>
mkdir -p .models/clean-model
for f in config.json generation_config.json model.safetensors.index.json \
         tokenizer.json tokenizer_config.json vocab.json merges.txt special_tokens_map.json; do
  [ -e "$SRC/$f" ] && ln -sf "$SRC/$f" .models/clean-model/
done
ln -sf "$SRC"/*.safetensors .models/clean-model/

python3 main.py --alg model_free_ptq --scheme W8A8 \
  --model .models/clean-model --save-dir <产物目录>
```

## 4. MoE-Quant（指针）

需要**数据校准**的 MoE GPTQ / 打包时走 `utils/MoE-Quant/`。工作目录、两阶段命令、档位术语、adapter 注册 → **`13-ext-moe-quant.md`**（此处不重复命令表）。

骨架提醒：在 `utils/MoE-Quant/` 下跑；优先 `ls tests/run_*.sh`；改档位须重跑 quant 再 pack（`metadata.pt` 会覆盖 pack CLI）。

## 5. 常用覆盖参数

| 参数 | 适用场景 |
|------|----------|
| `--needs-data true\|false` | 强制 / 跳过校准数据集 |
| `--dataset` | 校准数据路径（默认 `./datasets/ultrachat_200k`） |
| `--ignore` | 见下方"`--ignore` 语义"；llmcompressor / `model_free_ptq` 均为**整表替换** |
| `--pipeline sequential` + `--sequential-targets Linear` | 省显存（更慢） |
| `--device-map cpu` / `--device-map auto_offload` | 省显存（`auto_offload` 先落 CPU/磁盘，按需 onload） |
| `--offload-hessians` | GPTQ 类路径计算 hessians 时省显存 |
| `--patch-for-llmc` | `save_pretrained` 阶段 OOM |
| `--print-model` | 加载后打印模型结构 |
| `torchrun --nproc_per_node=N main.py …` | 多卡（llmcompressor；>100B 建议 N=2 或 4） |

### `--ignore` 语义（llmcompressor / `model_free_ptq`）

| 路径 | `--ignore` 行为 | 依据 |
|---|---|---|
| 默认 alg / llmcompressor（`FP8_DYNAMIC`、`W8A8`、`W4A16`…） | **整体替换**内置默认值 | `llmc_oneshot.py` |
| `--alg model_free_ptq` | **整体替换**，自有 `_ARCH_IGNORE` / `_DEFAULT_IGNORE` | `templates/model_free_ptq.py` |

两者默认列表**内容还不一样**（`model_free_ptq` 多 `mtp.pre_fc_norm_embedding` 等），
「默认忽略哪些层」必须看当次 log 的 `ignore=[...]`。CLI 显式写 `--ignore` 会把默认保护**全部丢掉**，
必须自己把需要的默认项一起写上。匹配走上游 `compressed_tensors/utils/match.py`；效果以产物 dtype 为准
（见 `04-verify-eval.md` 第 2 节）。常见约定：`re:` 前缀 + 需要整名时自加 `$`。

MoE-Quant 侧的 `--ignore` / `--force_int8` 见 `13-ext-moe-quant.md` §3，与上表不是同一套代码。

### `--save-dir` 省略时的落点

不是任意目录，而是按 alg 拼出来（日志会 `warning` 打出来）：

| alg | 默认落点 |
|---|---|
| 通用 / llmcompressor | `{--model}-{--scheme}` |
| `preprocess` | `{--model}-preprocessed` |

即 `--model /data/Qwen3-4B --scheme W8A8` → `/data/Qwen3-4B-W8A8`。
**默认值会写在 `--model` 同级目录**，所以 `--model` 是相对路径时产物也落在相对位置，容易找不到 —— 建议始终显式给 `--save-dir`。

## 6. 跑完立刻做什么

不得仅凭 `check_artifact` 报「量化完成」。**默认**必须：产物契约 + dtype（`04` §1–2）→ **再完成 `04` §3.2**（PPL_b/q、ΔPPL、ΔNLL），才可声称量化交付（见 `SKILL` 硬门禁）。仅当用户**明确**跳过评测时可停在工程产物。
方案与产物 dtype 不符时，回查 `--ignore` 与 `targets`。ΔNLL ≥ 0.1 按 `00` F1/F2 升级。

```bash
python3 main.py --lm-eval "pretrained=$SAVE,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.6"
python3 ./skills/scripts/report_ppl_delta.py --baseline-ppl <PPL_b> --quant-ppl <PPL_q>
```

## 7. examples 索引

以 `ls examples/` 为准（下表会过期，仅作导航）：

| 目录 | 说明 |
|------|------|
| `examples/deepseek/` `examples/qwen/` `examples/glm/` `examples/kimi/` | 各族脚本 / README |
| `examples/mimo/` `examples/minimax/` `examples/hy4/` | 同上 |
| `examples/preprocess/` | 预处理说明（若存在） |
