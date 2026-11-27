---
name: one-click-quant
description: >-
  one-click-quant 仓库的一键量化使用指南，覆盖量化历史适配判定、Dense/MoE/超大模型链路、
  已适配方案菜单、客户自提评估、未适配推荐、llmc/examples/MoE-Quant 三类路径、
  模型准备、量化执行、preprocess（QuaRot/AWQ/Smooth 可开条件）、scheme 扩展、
  评测验收（ΔNLL<0.1）、精度调优排查。
  当涉及 one-click-quant、新模型、未适配模型、已适配模型、量化方案推荐、客户自定义方案评估、
  模型量化方法推荐、method-model-map、preprocess 是否需要、main.py、llmc、examples、MoE-Quant、
  slimquant、--scheme、--alg、model_free_ptq、preprocess recipe、adapter、quant_templates.yaml、
  pack_quantized、build_mini、model_xray、torch_infer、check_artifact、check_recipe、
  report_ppl_delta、mixed_precision、PPL/lm_eval、ΔPPL、ΔNLL、auto_quantize、autoquantize_plan、
  effective-bits、ModelOpt、HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES 或精度调优时使用。
---

# one-click-quant

一键量化操作手册。先判「执行 / 扩展 / 知识」，再进对应文档。
参数不清楚时读 `python3 main.py --help`（`--scheme` help 不是全集，见 `03` §1）。
文档索引见 [README.md](README.md)。

## 客户给出模型时

1. **必须先读** [00-new-model-playbook.md](references/00-new-model-playbook.md)
2. 取证 → 判适配（**有量化历史=已适配**）→ 菜单 / 自提 / 未适配 Dense·MoE·超大链路（`00`）
3. 未适配或要定 scheme/alg 时读 [06-method-model-map.md](references/06-method-model-map.md)；
   **必须**跑 `skills/scripts/score_quant_methods.py`（分值：`score_rubric.yaml`；改分见 [07](references/07-match-score-rubric.md)）
   按匹配度列出候选（含优势/劣势）供客户选择
4. **客户确认方案前不得开跑量化**（跳过确认仅当用户明确声明，见检查清单）；未适配 >500 GB 须先提醒联系管理人员

## 执行

1. [01-env-docker.md](references/01-env-docker.md) — 环境 / 镜像 / 依赖
2. [02-model-prep.md](references/02-model-prep.md) — 权重取证 / mini / 解包
3. [03-quant-runbook.md](references/03-quant-runbook.md) — scheme / alg
4. [04-verify-eval.md](references/04-verify-eval.md) — 契约 / dtype / PPL
5. [05-precision-debug.md](references/05-precision-debug.md) — 精度排查（工具见 `20`）
6. [06-method-model-map.md](references/06-method-model-map.md) — **模型 → 方法对照 + 匹配度列表**
7. [07-match-score-rubric.md](references/07-match-score-rubric.md) — **匹配度打分规则**（`score_rubric.yaml`）

## 扩展

- preprocess → [10-ext-preprocess.md](references/10-ext-preprocess.md)
- scheme / 模板 → [11-ext-scheme.md](references/11-ext-scheme.md)
- MoE-Quant（含仅路由专家 · 混合 W4A8 术语）→ [13-ext-moe-quant.md](references/13-ext-moe-quant.md)

## 知识

- `--analysis`（事后敏感层诊断）→ [20-analysis.md](references/20-analysis.md)
- **AutoQuantize**（事前 INT4/INT8 位宽计划）→ [22-auto-quantize.md](references/22-auto-quantize.md)
- 易误读边界 → [experiences.md](references/experiences.md)

## 不负责 / 去向

| 本 skill 负责 | 不负责 |
|---|---|
| 适配判定、方案菜单、自提评估、未适配推荐（`00`） | 非 bug 改框架源码 → 上游 |
| 可执行命令与 scheme/alg 选择 | 集群调度、HF 账号、模型下载 |
| preprocess / MoE-Quant / scheme 扩展点 | 评测平台、推理服务运维 |
| 产物契约；精度报告（PPL_b/q、ΔPPL、ΔNLL）；**ΔNLL<0.1 才算精度正常**（`04`） | 「可行方案」≠ 自动精度合格 |
| AutoQuantize 计划生成 → `plan_to_scheme.py` → 按 `autoquantize_scheme.json` 量化（`22`） | 把 plan / Top-10 当已量化产物；跳过 scheme 转换；未注册 EP 却开 `--parallel ep` |
| 三类路径：llmc / examples / MoE-Quant；Dense·MoE·超大链路（`00`/`03`） | SlimQuant；超大走 llmc；Dense 超大强制量化 |

## 硬门禁（不满足即停）

- [ ] `--model` 真实存在且 `config.json` 可解析；用户未给则先问，**不得**编造路径
      - **必须相对路径**：`--alg model_free_ptq`（见 `03` §3）
      - 其它走 `cached_file` 的路径若报 `does not appear to have a file named`：先按 `03` §3 建相对软链
- [ ] 已完成 `00`：**已适配=有量化历史** / 未适配；未适配且 **>500 GB** 已提醒联系管理人员（坚持则按超大规则）
- [ ] 改 preprocess 前：`python3 ./skills/scripts/check_recipe.py <recipe>` 退出码 0
      （`<recipe>` = `preprocess/recipes/` 下本模型文件）；transform 须满足 `10` 可开条件
- [ ] 声称「**量化交付**」/「量化完成」前（**默认必须**，除非用户**明确**跳过评测）：
      1. `python3 ./skills/scripts/check_artifact.py <产物目录>` 退出码 0（`04` §1–2）
      2. 完成 `04` §3.2：给出 PPL_b / PPL_q / ΔPPL / ΔNLL，并报告是否 **ΔNLL < 0.1**
      - 用户明确跳过评测时：只能报「工程产物就绪（已跳过精度）」，**不得**暗示精度合格或完整交付
      - **不适用于** `--alg preprocess` 浮点产物（见 `10`）
- [ ] 声称「**精度合格**」前：必须已完成 `04` §3.2，且 **ΔNLL < 0.1**
- [ ] `--lm-eval` 前：`HF_DATASETS_CACHE`（默认 `./datasets`）下有本仓库 `lm_eval` 封装所需离线缓存
      （目录名见 `04` §3；此为仓库封装限制，非通用评测规范）

## 红线

- 不得跳过 `00` 适配判定（**量化历史**口径）直接开跑或凭模型名猜路径
- 已适配：不得隐瞒历史菜单；自提方案须先评估，不可行不得强制执行
- 未适配 >500 GB：不得跳过「联系管理人员」提醒；Dense 超大禁止未经适配强制套用 llmc
- 量化路径仅 llmc / examples / MoE-Quant；**禁止 SlimQuant**
- 不得为非本模型复用其它模型的 preprocess recipe；不得对不满足 `10` 条件的 transform 在不满足条件时强制启用
- 不得为绕过报错改 `absorb` / `pipeline` / `transform` / `quant_templates` 核心逻辑
- 不得伪造命令输出，或以「应该能跑」代替 exit code / 产物判定 / ΔNLL
- 用户未明确跳过评测时：不得在未完成 `04` §3.2 时声称量化交付/完成，或默认释放算力资源
- 不得在失败后换未文档化命令续跑以规避
- 不得混用两套 adapter 注册表（`preprocess/adapters/` ≠ `utils/MoE-Quant/src/models/registry.py`）
- MoE-Quant W4A8：术语与推荐纪律 **只引用** [`13` §3.1](references/13-ext-moe-quant.md)（此处不复述档位名）
- 运行前提见 `01`：`main.py` 量化须 Python ≥3.10；`skills/scripts/score_quant_methods.py` 可在更低版本跑打分（不得与量化执行所用解释器混用或等同）；目录可写；站在仓库根

## 检查清单

- [ ] 已完成 `00` 取证 + 适配结论（历史）+ Dense/MoE/超大分支 + 阶段 D（匹配度列表或历史菜单）
- [ ] **客户已确认方案编号**（唯一允许的例外：用户**书面/明确**写明「跳过方案确认并自行承担选错路径风险」；否则确认前不得开跑）
- [ ] 模型路径与 `config.json` 有效；所选路径属于三类之一且对该模型有入口
- [ ] 产物盘空间充足（约源权重规模）

## 通过 / 失败

- **通过（量化·交付）** = 契约齐全（`check_artifact`）+ 已完成 `04` §3.2；若 **ΔNLL < 0.1** 可同时报精度合格，否则须按 `00` F / `05` 升级或明确未达标
- **通过（仅工程·已跳过评测）** = 仅当用户**明确**跳过评测：契约齐全，且报告中写明「已跳过 `04` §3.2」
- **通过（preprocess）** = exit 0 + 浮点权重 + 无量化配置键（`10`）
- **失败** = 任一项不满足：报「命令 + 退出码 + 错误摘要」后停止；未做 §3.2 且用户未跳过 = 不得报交付完成

## 快速校验

```bash
python3 ./skills/scripts/check_recipe.py preprocess/recipes/<recipe>   # ls 该目录选真实文件
python3 ./skills/scripts/check_artifact.py <产物目录> --expect-quant-method <期望值>
python3 tools/model_xray.py --input_dir <模型目录> --output_dir <输出目录>   # 必须显式 --output_dir
python3 ./skills/scripts/report_ppl_delta.py --baseline-ppl <PPL_b> --quant-ppl <PPL_q>
python3 ./skills/scripts/plan_to_scheme.py --plan <autoquantize_plan.json> --out <autoquantize_scheme.json>
python3 ./skills/scripts/score_quant_methods.py --model <HF目录> [--target auto|fp8|int8|w4|w4a8]
```
`model_xray` 无非零退出码语义，读其 JSON。`<期望值>` 对照 `04` §1.1（多为 `compressed-tensors`）。AutoQuantize 每次跑完须生成 scheme（`22` §6）。
`score_quant_methods`：工程匹配度（非 ΔNLL）；改分见 `07`；推荐阶段按 `06` 展示。
