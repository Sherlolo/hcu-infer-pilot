# 模型方案入口：先判适配 → 已适配可选 / 客户自提评估 / 未适配新模型流程

> **何时用**：客户给出 HF 模型目录，要量化、要方案、或不确定是否已适配。
> **权威**：能力以仓库代码 + [`../README.md`](../README.md) 为准；命令细节见 `02`/`03`/`10`–`13`。
> **本阶段产出**：工程可行结论 + 推荐路径；精度是否合格以 `04` §3.2（**ΔNLL < 0.1**）为准，确认前不承诺精度。

## 本文件专属规则

仓库级硬门禁 / 红线 / 通过标准以 [`SKILL.md`](../SKILL.md) 为准（此处不重复）。本文件只补流程专属条：

1. **必须先读**客户真实模型目录（`config.json` + 权重索引/分片），禁止只凭模型名猜。
2. **必须先判定「已适配 / 未适配」**（定义见阶段 S），再分支；不得跳过判定直接开跑。
3. 客户确认方案编号前：**不得**开跑量化（跳过确认仅当用户明确声明，见 `SKILL` 检查清单）。
4. 「可行」= 当前仓库**已有入口或文档化可适配路径**；不要列理论上的任意 PTQ。
5. 规模门禁 / 超大 / F 升级 → 本文；MoE W4A8 术语 → **只引** `13` §3.1；transform → `10`；精度 → `04` §3.2；首选方案排序 → D + `score_quant_methods`。

## 总流程

```text
① 取证（A，含规模）→ ② 判适配（S）
        ├─ 已适配 → 方案菜单（T）→ 自提则评估（U）
        │              └─ ③ 主推荐（D）：历史置顶；可辅以 score_quant_methods
        └─ 未适配 → 规模门禁（Z）→ ③ 主推荐（D）：必须 score_quant_methods 按匹配度列表
                     （候选语义见阶段 B；对照表见 06）
        → ④ 客户确认方案编号
        → ⑤ 确认后执行（E）
        → ⑥ 工程失败或 ΔNLL≥0.1 才按 Dense/MoE 条件升级（F）；精度升级须客户同意
```

要点：**D 选定首选方案**（打分/菜单）；**F 只做升级**，不是确认前的默认长链路。

---

## 阶段 A：取证

| 来源 | 提取 |
|------|------|
| `<model>/config.json` | `model_type`、`architectures`、规模字段、MoE/MLA/MTP/VL 线索（含嵌套 `text_config`） |
| 权重 index 或单分片 | 键名模式；`weight_scale_inv` / `weight_packed` / per-expert / fused experts |
| 磁盘占用 | 模型目录总大小（`du -sb` 或等价）；用于 500 GB / 超大门禁 |
| `tools/model_xray.py --output_dir <非模型目录>` | dtype/shape；**禁止**把 xray JSON 写进模型目录 |
| 仓库对照 | **量化历史**（见阶段 S）；另扫 `ADAPTER_REGISTRY`、`preprocess/recipes/`、MoE-Quant registry、`quant_templates.yaml`、`examples/<family>/` 作能力辅助证据 |

结构标签（打勾）：权重 `bf16_fp16`/`block_fp8`/`mxfp4`/`already_quantized`；拓扑 `dense`/`moe`/`mla`/`linear_attn`/`mtp`/`vl`；expert `per_expert_2d`/`fused_3d`/`unknown`；规模 `≤500GB` / `>500GB`；仓库 `has_quant_history`/`has_preprocess`/`has_moe_quant`/`has_scheme_yaml`/`has_examples`/`none`。

取证摘要须含：路径、架构、**dense/moe**、规模（约 GB）、权重形态、是否有量化历史及命中证据。

---

## 阶段 S：是否「已适配量化」

**定义（本 skill 唯一口径）**：

| 结论 | 含义 |
|------|------|
| **已适配量化** | 仓库内存在**可写出具体路径/文件名**的量化历史（见下）；证据须点名本模型或客户确认的同结构变体 |
| **未适配量化** | 无上述可复现历史 |

**量化历史**证据（满足任一条即可；输出时必须抄出路径）：

- `examples/<family>/` 下脚本或 README **点名**该模型（或客户书面确认可用的同 `architectures` + 同专家/层布局变体）
- MoE-Quant `tests/run_*` / 文档化 runbook **点名**该模型且可复跑
- 仓库内该模型的量化产物说明、scheme/配方快照、或已合并的定制转换链路记录

**不算**量化历史：仅「通用可试」（`FP8_DYNAMIC` / `W8A8` / `model_free_ptq`）；仅有 preprocess recipe；仅族名相近（如都叫 Qwen）但脚本未点名、客户未确认变体。

能力辅助证据（adapter / YAML / registry）用于**菜单怎么写**，**不能单独**把模型判成已适配。  
「同结构变体」默认**不算**已适配，除非客户确认变体范围，或历史文件已写明覆盖该变体。

```bash
# 历史与能力对照（以实际文件为准）
ls examples/
find examples -iname '*<model_hint>*' | head
ls utils/MoE-Quant/tests/run_* 2>/dev/null | head
ls preprocess/recipes/
```

输出：`适配结论：已适配量化（历史：…）| 未适配量化`。

---

## 阶段 Z：未适配规模门禁（≤500 GB）

仅对 **未适配量化** 生效：

1. 模型目录 **≤ 500 GB**：可按阶段 F 的 Dense/MoE 默认链路推进。
2. 模型目录 **> 500 GB**：
   - **必须先提醒**：请客户优先联系 one-click-quant 管理人员。
   - 客户书面/明确坚持继续 → 按 **超大模型** 规则（下文）处理，**不得**再走普通 Dense·llmc 路径。

> **500 GB** 是本 skill / 本仓库流程门禁（与 `score_rubric.yaml` 的 `gates.dense_ultra_gb` 对齐），不是物理定律；对外说明时写清「本流程规则」。

---

## 超大模型量化规则

适用：未适配且目录 **>500 GB**，且客户在提醒后仍要求量化；或管理人员明确按超大受理。

1. **禁止 llmc**（`main.py` / llmcompressor / `model_free_ptq` 通用路径）。
2. **Dense 超大：本流程不支持**。拓扑为 dense 时明确告知；禁止未经适配强制套用 llmc；不得将 Dense 误按 MoE 路径处理。
3. **MoE 超大：阶段 D 确认首选方案后，失败/精度不达标再按 F2 升级**（preprocess / MoE-Quant / analysis）。

---

## 阶段 F：Dense / MoE 默认量化链路（未适配或无历史菜单时）

> 已适配时优先复用历史菜单（T）；下列为**未适配**或历史不可用时的**条件升级**（默认停在第①档（初始方案），不是必跑完整升级序列）。
> 首选方案已在阶段 D 由客户确认（匹配度 Top / 历史菜单）；本阶段只处理**失败或精度不达标后的升级**。
> 同档修复（如 VL 改 `--alg gptq`、OOM 加 sequential）仍属第①档（初始方案）。工程成功后再做 `04` §3.2；ΔNLL ≥ 0.1 或硬失败才升级。精度升级档须客户同意时间预算。`05` 只引用本文，不另立升级序列。

### F1. Dense 大模型

```text
① 已确认的匹配度 Top / 菜单方案开跑（含同档修复：gptq / sequential / offload）
    ├─ 工程成功 → 04 §3.2
    │     ├─ ΔNLL < 0.1 → 结束
    │     └─ 不达标 →（客户同意精度升级）② QuaRot（须 10 条件；Dense 不默认开 AWQ/Smooth）
    │           → 再量化 → 04
    │                 └─ 仍不达标 → ③ --analysis 回退 → 再量化 / 叫停
    └─ OOM/硬失败且同档修复无效
          → ② AutoQuantize → plan_to_scheme → 落地 examples 风格脚本 → 量化 → 04
                └─ 不达标 → 同上 QuaRot / analysis（客户同意）
```

要点：

- 未 OOM / 未硬失败前不得跳到 AutoQuantize；同档修复不算下一档。
- AutoQuantize 只出 plan/scheme（`22`），**必须**落到可执行脚本后再量化。
- Dense 精度升级 preprocess **仅 QuaRot**（条件见 `10`）。

### F2. MoE 大模型

```text
① 已确认的匹配度 Top：examples 强命中 → 脚本；否则多为 AutoQuantize 落地脚本（一步）
    → 04 §3.2
         ├─ ΔNLL < 0.1 → 结束
         └─ 不达标（客户同意）→ ② preprocess（按 10 条件选 QuaRot / AWQ / Smooth）
              → 再量化 → 04
                   └─ 仍不达标 → ③ MoE-Quant（registry 命中）→ 04
                        └─ 仍不达标 → ④ --analysis 回退 / 叫停
```

要点：

- 无 examples 强命中时，禁止进行无效检索后转入 AutoQuantize；首选方案以阶段 D 确认结果为准。
- preprocess 的 transform **按条件自适应**（`10`），不是固定全开。
- MoE-Quant 属精度升级路径（或历史菜单本身即为此路径），不得在未经匹配度确认时作为默认首选。

---

## 阶段 T：已适配 — 菜单

只列历史记录对应的路径 + 仍适用的辅助入口。每条写清：方案名、是否 preprocess、命令骨架、预期 `quant_method`、限制（相对路径、MoE 档位见 `13` 等）。

客户选编号 → 确认后进 `02`/`03`；自提 → U；既不选也不提 → 追问一次后停。

---

## 阶段 U：自提方案评估

逐条结论：

1. 目标格式是否支持（MoE W4A8 档位见 `13` §3.1）；是否落在三类路径之一（禁止 SlimQuant）
2. 是否匹配当前结构（MoE / fused / FP8 / 相对路径等）及规模门禁
3. 有无入口；无则「小改」（ignore/recipe/YAML/examples）还是「大改」（新 adapter/新脚本）
4. 是否建议 preprocess（条件见 `10`；Dense 精度升级优先 QuaRot）
5. 相对已适配菜单的差异与风险

三态：**可行（直接跑）** / **可行（需小改）** / **不可行**（回流菜单或给替代）。确认前不执行。

---

## 阶段 B：路径 ID 词典（辅助词典；排序以 D 打分为准）

对菜单/打分结果里出现的路径写清语义。**推荐次序以 `score_quant_methods` / 历史菜单为准**，本表不规定「MoE 优先 examples」等固定次序。无命中证据则不得列入推荐。须服从超大门禁与 F 升级条件。

| ID | 路径 | 列入条件 | 要点 |
|----|------|----------|------|
| L1 | llmc（`main.py` scheme / ptq / model_free_ptq） | Dense；非超大 | 打分常落于此；OOM/硬失败再按 F1 降级 |
| E1 | `examples/<family>/` 脚本 | 强命中近亲脚本；或 AQ 落地后的脚本 | 非统一 `main.py`；弱族名命中不得作为首选方案 |
| A0 | AutoQuantize → `plan_to_scheme` → 再写脚本 | 能分析或已有 EP；再量化 | 只出 plan；见 `22` |
| Me | MoE-Quant **仅路由专家 W4A8** | MoE；`--quantize_only_experts` | 见 `13` §3.1 |
| M3 | MoE-Quant **混合 W4A8** | MoE；force_int8 或固定配方 | 见 `13` §3.1 |
| M1 | MoE-Quant W4A16 | MoE；F2 或历史菜单 | 见 `13` |
| M2 | MoE-Quant W8A8（`--bits 8`） | MoE；工程可做 | **不是**「纯/裸 W4A8」；见 `13` |
| G3 | `--scheme FP8_TO_BF16` | block FP8 等 | 解包，非量化 |
| P→Q | preprocess 后再走 L1/E1/M* | 满足 `10` 对应 transform 条件 | Dense 精度升级优先 quarot |

> **SlimQuant** 不列入。MoE W4A8 术语与推荐纪律 **只引用** [`13` §3.1](13-ext-moe-quant.md)，此处不复述。

---

## 阶段 C：是否 preprocess

先查 `10` 的 **可开条件**；不满足则标「不可开该 transform」，禁止在不满足条件时强制启用。

| 信号 | 倾向 |
|------|------|
| Dense：llmc/脚本后 ΔNLL ≥ 0.1，且 QuaRot 条件满足 | **建议 quarot** |
| MoE：脚本后 ΔNLL ≥ 0.1 | 按条件在 quarot / AWQ / Smooth 中自适应选择 |
| 已有同模型/客户确认变体的 preprocess（`has_preprocess`）且 `10` 条件仍满足 | **建议**复用该 recipe（须先 `check_recipe`） |
| 只要快速试通、尚未量化 | 可选 / 先不建议 |
| 已量化或只要解包（G3） | 不建议 |
| 条件不满足（见 `10`） | **跳过该 transform**，改其它精度升级档 |

输出一句：**建议 | 可选 | 不建议 | 有限** + 拟用 transform + 理由 + recipe 状态（须可执行，禁止使用无法验收的程度副词）。

---

## 阶段 D：主推荐（按匹配度列出供客户选）

> 总流程第③步。本阶段产出**供确认的首选方案列表**；F 不在此展开。

优先级：

1. 已适配 → 历史菜单置顶（T）；其余方法仍可作为辅助对照（可跑打分）
2. 未适配 → 规模门禁（Z）已过后：Dense/MoE/超大只约束**候选集合**，**排序以打分为准**
3. 未适配（及需要辅助对照时）**必须**用 [06-method-model-map.md](06-method-model-map.md) + `score_quant_methods.py`（分值见 `score_rubric.yaml` / [07](07-match-score-rubric.md)）打出匹配度表：

```bash
python3 ./skills/scripts/score_quant_methods.py --model <HF目录> [--target auto|fp8|int8|w4|w4a8]
```

4. 向客户展示：**按匹配度降序**的方法列表，每项含优势、劣势、命令骨架（模板见 `06` §0）；不得只给单一默认方案
5. 资源提示（MoE-Quant EP：`num_experts % nproc == 0`；AutoQuantize 单卡/EP）写入对应项的「劣势/限制」
6. 精度闭环：写明「量化 → `04` → ΔNLL < 0.1 否则按 F 升级」；并声明匹配度 ≠ 精度保证

客户选定编号后，再写完整命令与产物目录。确认前不执行。

---

## 阶段 E：确认之后

1. 只改文档指定扩展点；preprocess 先 `check_recipe.py`
2. 执行：`01` → `02` → `03`；MoE 见 `13`；链路升级见 F1/F2
3. 契约：量化用 `check_artifact.py`；preprocess 浮点按 `10`
4. 精度：默认必须完成 `04` §3.2 才可报量化交付（用户明确跳过除外，见 `SKILL`）；**ΔNLL ≥ 0.1 不得报精度合格**，继续 F 升级或 `05`

---

## 与其它文档

| 文档 | 职责 |
|------|------|
| **`00`（本文）** | 总流程；适配；Z/超大；**D 选定首选方案**；**F 升级**；菜单/自提 |
| **`06`** | 强制打分展示 + scheme 语义 + 导航快照 |
| **`07`** | 打分开发文档（改分 / key / 冒烟） |
| `02`/`03` | 确认后的准备与命令（不另行规定首选方案优先级） |
| `10` | preprocess + transform 可开条件 |
| `11`/`13` | scheme 扩展；**MoE W4A8 术语唯一正文** |
| `04`/`05` | 契约、ΔNLL；精度排查（回退表=经验） |
| `20`/`22` | `--analysis` / AutoQuantize |
| `SKILL` | 仓库级硬门禁 / 红线 / 通过标准 |
