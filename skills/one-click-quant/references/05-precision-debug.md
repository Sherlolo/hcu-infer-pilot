# 阶段5 精度排查

> **权威来源**：仓库根 `docs/precision-tuning.md`；分析工具 `20-analysis.md`。
> **升级阶梯以 `00` 阶段 F / 超大规则为准**（本文不另起一套）。精度门槛见 `04` §3.2。

## 何时进本文

产物已过 `04` 契约 / dtype，但精度未达标（或客户要继续降低误差）。先确认评测可比（同结构、同 wikitext 封装参数），再：

| 类型 | 升级顺序 |
|------|----------|
| Dense | `00` F1 |
| MoE | `00` F2 |
| 超大 | `00` 超大规则（禁 llmc；Dense 超大不支持；MoE 走 F2） |

测 PPL / 算差：

```bash
python3 main.py --lm-eval "pretrained=<量化产物真实目录>,tensor_parallel_size=2,dtype=auto,gpu_memory_utilization=0.6"
python3 ./skills/scripts/report_ppl_delta.py --baseline-ppl <PPL_b> --quant-ppl <PPL_q>
```

做 ignore 的 A/B 时评测参数必须不变。

## 定位：用 `--analysis`，不用猜

```bash
python3 main.py --model <真实模型目录> --save-dir <量化产物目录> --analysis mse
python3 main.py --model <真实模型目录> --save-dir <量化产物目录> \
  --analysis activation_sensi --num-calibration-samples 128 --max-seq-length 512
```

五种模式与列含义见 `20`。把最差层加入回退集合后重新量化，再测精度。

**ignore**：llmcompressor / `model_free_ptq` 路径为**整表替换**（须自带 `lm_head` / gate / embed）。见 `03` §5。

位宽分配问题（该 W4 还是 W8）优先回 **AutoQuantize**（`22`），不要只靠事后 ignore。

## 回退候选速查（经验起点，非规则）

**必须**先跑 `--analysis`；下表不得单独作为 ignore/回退依据。与分析冲突时**以分析为准**。一次只动一类，再测 `04` §3.2。

| 类别 | 经验起点（须被 analysis 印证） |
|---|---|
| `mlp.down_proj` / `self_attn.o_proj` | 常出现在敏感层列表前列 |
| MoE `gate` / `router` | 常见默认 ignore |
| MoE `experts` | 常见低比特主落点 |
| `shared_experts` / MLA 低秩投影 | 常保持高精度 / 条件排除 |
| `lm_head` / `embed_tokens` / `*norm` | 常见默认 ignore |

## 预处理 A/B 与 OOM

- 只开满足 `10` 可开条件的 transform；Dense 精度升级仅 QuaRot
- 真实激活：`--act-stats` + recipe **对应 transform 块下** `use_calib_acts: true`
- 校准 OOM：`--pipeline sequential`、`--offload-hessians`、调小 batch/seq；保存 OOM：`--patch-for-llmc`
- Dense llmc 持续 OOM：按 `00` F1 切 AutoQuantize + examples 脚本
