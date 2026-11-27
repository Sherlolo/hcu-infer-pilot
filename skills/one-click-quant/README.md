# one-click-quant Skills

本目录是仓库的一键量化操作手册，按「执行 → 扩展 → 知识」组织。

> 编号：`0x` 生命周期（`00` 入口 → `01`–`07` 执行），`1x` 扩展（10→13，`12` 空缺），`2x` 知识（20、22；`21` 空缺）；
> `experiences.md` 不占编号。硬门禁 / 红线 / 通过标准见 [SKILL.md](SKILL.md)。

客户给出模型时的入口与硬门禁见 [SKILL.md](SKILL.md) / [00-new-model-playbook.md](references/00-new-model-playbook.md)。
定「该用什么量化方法」见 [06-method-model-map.md](references/06-method-model-map.md)；
改匹配度分值见 [07-match-score-rubric.md](references/07-match-score-rubric.md) / [scripts/score_rubric.yaml](scripts/score_rubric.yaml)。

## 文档导航

### 执行

| 文档 | 何时读 |
|------|--------|
| [00-new-model-playbook.md](references/00-new-model-playbook.md) | 总流程；适配；D 选定首选方案；F 升级；菜单/自提 |
| [01-env-docker.md](references/01-env-docker.md) | 环境；Python：量化 ≥3.10 vs 打分脚本 |
| [02-model-prep.md](references/02-model-prep.md) | 权重取证、建 mini、反量化 |
| [03-quant-runbook.md](references/03-quant-runbook.md) | 确认后的命令（不另行规定首选方案优先级） |
| [04-verify-eval.md](references/04-verify-eval.md) | 产物契约 / dtype / PPL；ΔPPL / ΔNLL |
| [05-precision-debug.md](references/05-precision-debug.md) | 精度排查（回退表=经验，须 analysis） |
| [06-method-model-map.md](references/06-method-model-map.md) | 强制打分展示 + scheme 语义 + 导航快照 |
| [07-match-score-rubric.md](references/07-match-score-rubric.md) | 打分开发文档（数字在 `score_rubric.yaml`） |

### 扩展

| 文档 | 何时读 |
|------|--------|
| [10-ext-preprocess.md](references/10-ext-preprocess.md) | preprocess adapter + recipe |
| [11-ext-scheme.md](references/11-ext-scheme.md) | `--scheme` / 模板 |
| [13-ext-moe-quant.md](references/13-ext-moe-quant.md) | MoE-Quant 接入；仅路由专家 · 混合 W4A8 |

### 知识

| 文档 | 何时读 |
|------|--------|
| [20-analysis.md](references/20-analysis.md) | `main.py --analysis` 敏感层诊断 |
| [22-auto-quantize.md](references/22-auto-quantize.md) | AutoQuantize：限制 / 用法 / 适配 / 读 plan / 制定方案 |
| [experiences.md](references/experiences.md) | 易误读边界（不重复正文流程） |

## 脚本

| 脚本 | 用途 |
|------|------|
| [scripts/check_recipe.py](scripts/check_recipe.py) | 校验 preprocess recipe |
| [scripts/check_artifact.py](scripts/check_artifact.py) | 校验量化产物契约 |
| [scripts/report_ppl_delta.py](scripts/report_ppl_delta.py) | 由 PPL_b / PPL_q 打印 ΔPPL / ΔNLL；标注是否 ΔNLL < 0.1 |
| [scripts/plan_to_scheme.py](scripts/plan_to_scheme.py) | AutoQuantize plan → `autoquantize_scheme.json`（见 `22` §6） |
| [scripts/score_quant_methods.py](scripts/score_quant_methods.py) | 读 `score_rubric.yaml` 计算匹配度，输出优劣表 |
| [scripts/score_rubric.yaml](scripts/score_rubric.yaml) | **匹配度分值唯一来源**（说明见 `07`） |
