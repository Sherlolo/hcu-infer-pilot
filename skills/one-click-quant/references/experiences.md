# 经验条目（追加制）

> 本表只沉淀**易误读边界**；流程以各 reference 为准，冲突时 reference 优先。
> 与正文重复的结论只保留指针，不重写全文。

## 追加规范

- 连续编号 `[E-序号]`，只追加不改号；证伪时新条标注「取代 E-xx」
- 来源：`实测` / `用户反馈` / `代码实证`；状态：`已回归` / `待验证` / `已上流 docs`

| 条目 | 主题 | 适用条件 | 结论要点 | 来源 | 验证状态 |
|---|---|---|---|---|---|
| E-01 | 镜像引用 | 起容器 | 支持 `registry/image:tag` 与本机 image ID；ID/`sha256:` 跳过 pull，且不可移植 | 用户反馈 | 待验证 |
| E-02 | 模型路径 | 任何 `--model` | 不扫盘、不编造。`model_free_ptq` 必须相对路径；其它 alg 遇 `does not appear to have a file named` 先按 `03` §3 建软链 | 实测 | 已回归 |
| E-03 | `--ignore` 语义 | 改忽略层 | llmcompressor / `model_free_ptq` 为**整表替换**（须自带 lm_head/gate/embed）。详见 `03` §5 | 代码实证 | 已回归 |
| E-04 | 容器问题 | 容器内失败 | 无通用解：按报错分类；禁无脑复跑固定 `docker run` | 用户反馈 | 待验证 |
| E-05 | analysis 数据集 | `--analysis` | 行为因模式而异（`fisher` 无回退；`activation_sensi` 可静默换数据）。详见 `20` | 代码实证 | 已回归 |
| E-07 | PPL 数据集 | `--lm-eval` | 本仓库封装需离线 wikitext 缓存；校准 ultrachat 不能替代。见 `04` §3 | 代码实证 | 已回归 |
| E-08 | 两套 adapter | 接新模型 | `preprocess/adapters/` ≠ MoE-Quant `registry.py`；后者匹配顺序敏感。见 `13` | 代码实证 | 已回归 |
| E-09 | 减层验证 | 建 mini | 「减层」≠「取前 N 层」：须覆盖该模型真实结构单元。见 `02` | 代码实证 | 已回归 |
| E-10 | 产物判定 | 量化是否成功 | 工程看 `quant_method` + dtype + 分片；**交付**还须 `04` §3.2（除非用户明确跳过）。见 `04`/`SKILL` | 代码实证 | 已回归 |
| E-11 | `ignore` 匹配 | 写 `--ignore` | 上游 match 语义以产物 dtype 为准，勿假设子串包含。见 `03` §5 | 代码实证 | 已回归 |
| E-13 | W4A8 两档术语 | MoE-Quant | 术语与推荐纪律见 `13` §3.1（此处不复述） | 代码实证 | 已回归 |
| E-14 | pack CLI 被覆盖 | MoE 打包 | ② 从 `metadata.pt` 覆盖同名 CLI；改档位须重跑①。见 `13` §2 | 代码实证 | 已回归 |
| E-15 | build_mini 残留配置 | mini 交付 | 多数脚本不 `pop("compression_config")`，交付前须删或按 `10` 验收 | 代码实证 | 已回归 |
| E-16 | model_xray 副作用 | 取证 | 必须显式 `--output_dir`；无非零退出码。见 `02`/`SKILL` | 代码实证 | 已回归 |
| E-17 | W4A8 历史入口 | 有 MoE-Quant 量化历史时 | 查 `ls utils/MoE-Quant/tests/run_*w4a8*.sh`；特例以 adapter `validate_*` 为准。见 `13` §3.3 | 代码实证 | 已回归 |
