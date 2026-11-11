# HCU-Infer-Pilot

`HCU-Infer-Pilot` 是一套面向 LLM Serving、性能分析、线上故障排查、模型优化历史检索和 DCU/ROCm kernel 经验沉淀的 Codex Skills 仓库。它把常见的 AI Infra 工作流拆成可复用的 skill、知识库和辅助脚本，方便你在同一个仓库里完成 benchmark、profiling、PR 分析、历史查询和优化闭环。

这个仓库适合下面几类工作：

- 比较 SGLang、vLLM、TensorRT-LLM 在同一模型、同一 workload 下的部署效果
- 分析 `torch.profiler` trace，定位 kernel、overlap、fusion 和 bubble 问题
- 查询模型架构图、社区 PR 优化历史、本地提交历史
- 排查 SGLang 线上服务异常
- 围绕一个模型持续做 SGLang SOTA 优化闭环
- 为 LightOp / DCU / ROCm 场景查找 kernel 证据、profiling 证据和下一步优化方向
- 为新模型、TP/EP 组合和 gfx 架构收集并调优 AITER ASM fused-MoE 配置
- 抓取模型真实 GEMM，对 hipBLASLt 与 NMZ rocBLAS 做 token M 调优、最近兼容点配置补齐和单算子精度/性能验证

## 仓库内容

仓库大体分成三类内容：

1. **直接可用的 skills**：可以作为对话指令使用，比如 `$llm-torch-profiler-analysis`。
2. **知识库型内容**：如 `model-pr-optimization-history`、`model-pr-optimization-hg-history`，用于为优化判断提供历史证据。
3. **辅助脚本与测试**：用于生成、校验、查询和回归测试这些 skills。

## Skills 总览

下表汇总当前仓库的 **25 个独立 Skill / 知识入口**，以及安装脚本引用的 **6 个 Humanize 辅助 Skill**。名称链接指向对应使用说明；子模块辅助技能需先初始化 `skills/lightop-skills`。下方保留常用技能的详细说明和提示词示例。

| 类别 | Skill / 入口 | 大概功能 | 主要产物或适用场景 |
| --- | --- | --- | --- |
| 模型服务 | [llm-serving-auto-benchmark](skills/llm-serving-auto-benchmark/SKILL.md) | 在相同模型、负载、卡数与 SLA 下比较 SGLang、vLLM、TensorRT-LLM 等框架的部署参数与性能。 | 候选启动命令、参数检查、benchmark 对比及失败记录。 |
| 模型服务 | [sglang-prod-incident-triage](skills/sglang-prod-incident-triage/SKILL.md) | 按现场采集、请求回放、工具升级的顺序诊断服务异常。 | 健康检查、排队、超时、hang、crash、输出及 PD/EP/HiCache 问题的诊断证据。 |
| 模型优化 | [sglang-sota-humanize-loop](skills/sglang-sota-humanize-loop/SKILL.md) | 组织公平 benchmark、领先框架 Prof 对比、SGLang patch 与 Humanize/RLCR 模型级迭代。 | 一个模型的性能差距、修复与复测闭环；需要实际运行环境。 |
| Prof 分析 | [llm-torch-profiler-analysis](skills/llm-torch-profiler-analysis/SKILL.md) | 分析已有 torch.profiler trace，或驱动在线服务分阶段采集。 | kernel、overlap、fusion、空泡和源码关联报告；支持 SGLang、vLLM、TensorRT-LLM 的相应采集条件。 |
| 模型理解 | [model-architecture-diagram](skills/model-architecture-diagram/SKILL.md) | 查询 LLM、VLM、MoE 等模型的公开原始架构图。 | 图片 Markdown、来源；无匹配时返回 `no_match`。 |
| 模型仿真 | [model-compute-simulation](skills/model-compute-simulation/SKILL.md) | 按模型配置和并行/输入形状展开算子计算模板，估算 FLOPs、tensor shape，并结合实测延迟估计 MFU。 | 算子计算量与并行方案比较；不代替真实性能测量。 |
| 量化 | [one-click-quant](skills/one-click-quant/SKILL.md) | 指导 one-click-quant 工程的量化、预处理、新模型 recipe、scheme/template 扩展与精度排查。 | 量化运行方案、适配方法、PPL/lm_eval 精度诊断。 |
| PR 分析 | [vllm-pr-summary](skills/vllm-pr-summary/SKILL.md) | 读取 vLLM PR 的描述、diff、讨论和 review，解释实现及风险。 | 含 Mermaid 图的中文 PR 总结报告。 |
| 历史知识 | [model-pr-optimization-history](model-pr-optimization-history/SKILL.md) | 检索开源社区中的模型及框架级优化 PR 历史。 | fast path、fusion、overlap、attention、MoE、cache 等改动出处与回归提示。 |
| 历史知识 | [model-pr-optimization-hg-history](model-pr-optimization-hg-history/SKILL.md) | 检索本地 SGLang/vLLM fork 的优化提交，与社区 PR 证据互补。 | 本地实现路径、提交和适用条件；SGLang 历史按 local-only 规则维护。 |
| 知识管理 | [kb-search](skills/kb-search/SKILL.md) | 在本地 fork 历史、社区 PR 历史、HCU 知识库及现有 Skill 之间路由查询。 | 可打开的资料出处、历史记录或技能入口。 |
| 知识管理 | [kb-ingest](skills/kb-ingest/SKILL.md) | 判断资料应进入 HCU 知识库、历史更新流程还是本仓可复用 Skill。 | 分类结果与对应入库流程，避免直接改写受保护内容。 |
| 知识管理 | [kb-update](skills/kb-update/SKILL.md) | 按各自规则更新 fork 提交历史、社区 PR 历史及 HCU 知识库的 SGLang PR 同步。 | 增量更新、校验和待审变更；发布与合并遵从该技能的授权规则。 |
| GEMM 调优 | [gemm-tuning](skills/gemm-tuning/SKILL.md) | 从 SGLang/vLLM 日志提取 hipBLASLt/rocBLAS shape，生成 tuning 输入并验证模型端命中。 | 调优输入、配置、运行时命中证据和中文报告。 |
| GEMM 调优 | [hipblaslt-rocblas-tuning](skills/hipblaslt-rocblas-tuning/SKILL.md) | 抓取真实 GEMM，按 token M 幂次点调优并用最近兼容点补齐整数尺寸。 | hipBLASLt override / NMZ rocBLAS native 配置、来源映射和单算子精度/性能验证。 |
| MoE 调优 | [hcu-aiter-moe-tuning](skills/hcu-aiter-moe-tuning/SKILL.md) | 从真实模型与 TP/EP 配置收集 AITER ASM fused-MoE shape，调优并校验配置表。 | gfx/量化相关的 CSV、覆盖检查、数值与服务验证。 |
| LightOp | [lightop-kernel-knowledge](skills/lightop-kernel-knowledge/SKILL.md) | 优先检查本地 LightOp 实现，再检索 ROCm/HIP/库及框架证据。 | 实现、测试、dispatch、配置和跨平台迁移依据。 |
| LightOp | [lightop-kernel-agent-loop](skills/lightop-kernel-agent-loop/SKILL.md) | 依据 HIP 性能证据新增或优化 LightOp 算子，完成构建、正确性、计时、ISA 与消融归因。 | 可复现、可回退的算子实现和验证记录；本仓独立入口直接驱动证据闭环。 |
| DCU Prof | [dcu-profiler-report](skills/dcu-profiler-report/SKILL.md) | 从 hipprof 入手，必要时深入 counters、ROCm/DTK profiler 与 ISA，解释 kernel 瓶颈。 | API/kernel/memory 证据和下一步具体算子修改建议。 |
| 模型/Kernel 诊断 | [kernel-model-triage](skills/kernel-model-triage/SKILL.md) | 对 HCU 训练/推理任务进行模型到 kernel 的跨层测试、性能及精度问题定位。 | 可比基线、OOM/编译/多卡问题诊断、回归与优化验收方案。 |
| HIP 基线 | [kernel-hip-baseline](skills/kernel-hip-baseline/SKILL.md) | 将 Torch、Triton、Python 或 CUDA/C++ 参考实现转换为 HCU HIP/C++ baseline。 | 可编译 `kernel.hip` 与正确性 harness，作为后续优化起点。 |
| HIP 优化 | [kernel-hip-optimize](skills/kernel-hip-optimize/SKILL.md) | 通过 hipprof、roofline、ISA、分支探索及消融归因迭代优化 HIP/CK Tile kernel。 | 固定 shape 的候选、正确性、性能与经验记录。 |
| HIP 方法 | [kernel-hip-handbook](skills/kernel-hip-handbook/SKILL.md) | 提供 BW 系列 HCU 的指令、LDS、寄存器、occupancy、roofline 和 shape 特化方法。 | 深度分析与优化参考，按实际架构和测量选用。 |
| Triton 基线 | [kernel-triton-baseline](skills/kernel-triton-baseline/SKILL.md) | 从参考实现与 shape 生成正确性优先、benchmark 兼容的 Triton baseline case。 | `kernel.py`、`ref.py`、manifest 和基线验证；复杂算子需要专门实现。 |
| Triton 优化 | [kernel-triton-optimize](skills/kernel-triton-optimize/SKILL.md) | 联合固定配置计时、profiling、KB、Triton IR/cache、ISA 和消融进行迭代优化。 | 多架构 HCU Triton kernel 的候选对照与归因报告。 |
| 辅助研究（子模块） | [ask-codex](skills/lightop-skills) | 通过 Humanize 运行时调用 Codex 辅助研究或复核。 | 辅助意见；依赖对应工具和运行时配置。 |
| 辅助研究（子模块） | [ask-gemini](skills/lightop-skills) | 通过 Humanize 运行时调用 Gemini 辅助研究。 | 辅助意见；依赖对应工具和运行时配置。 |
| 协作流程（子模块） | [humanize](skills/lightop-skills) | 提供 Humanize 协作与迭代工作流入口。 | 配合计划、实施和复核使用的运行时流程。 |
| 计划（子模块） | [humanize-gen-plan](skills/lightop-skills) | 为任务生成实施计划。 | 可供后续细化和执行的计划。 |
| 计划（子模块） | [humanize-refine-plan](skills/lightop-skills) | 复核并细化已有计划。 | 修订后的步骤、约束和验证安排。 |
| 迭代复核（子模块） | [humanize-rlcr](skills/lightop-skills) | 组织 Humanize/RLCR 实施与复核循环。 | 有检查和停止规则的迭代记录。 |

**安装覆盖说明：** 当前 `install_all_skills.sh` 注册了 23 个安装项，并不自动安装表中所有独立目录。`gemm-tuning`、`one-click-quant` 及六个 `kernel-*` 入口可按“方式二”单独安装。本仓已有独立版本的 `lightop-kernel-knowledge`、`lightop-kernel-agent-loop` 和 `dcu-profiler-report`，但一键安装器目前仍从 `lightop-skills` 子模块或其 Humanize 运行时取对应内容；使用时核对实际安装来源，不能将它们视为同一版本。两个历史知识目录的 frontmatter 均使用 `model-pr-history-knowledge`，引用时同时说明目录，以免混淆社区 PR 与本地 fork 历史。

### 1. `llm-serving-auto-benchmark`

作用：跨框架 LLM Serving benchmark。适合比较 SGLang、vLLM、TensorRT-LLM 或其他 serving 框架在同一模型、同一 workload、同一 GPU 预算下的表现。

典型能力：

- 基于 cookbook 或本地配置生成候选启动命令
- 校验框架 CLI 参数是否真实存在
- 汇总 benchmark 结果并输出对比表
- 保留失败/跳过候选，便于复盘

推荐提示词：

```text
$llm-serving-auto-benchmark
工作目录是：<路径> 工作容器是：<容器名>
请针对模型 <模型名>，在 <GPU型号/数量> 和 <SLA> 约束下，对比 SGLang、vLLM 的最佳部署参数。
请先确认可用 cookbook / 启动命令，再设计候选搜索空间，最后输出可执行的 benchmark 方案和结果汇总格式。
```

### 2. `llm-torch-profiler-analysis`

作用：统一分析 LLM Serving 的 `torch.profiler` trace，支持 SGLang、vLLM

典型能力：

- 分析已有 `trace.json` / `trace.json.gz`
- 对 live server 进行 profiling
- 输出 kernel、overlap-opportunity、fuse-pattern、bubble-gap 分析
- 必要时生成模型结构图并嵌入报告

推荐提示词：

```text
$llm-torch-profiler-analysis
工作目录是：<路径> 工作容器是：<容器名>
请分析这个 trace 目录 / 文件：<trace 路径>。
模型是 <模型名>，框架是 <SGLang/vLLM/TensorRT-LLM>。
请输出一份完整的中文 triage 报告，包含 kernel、overlap、fuse-pattern、bubble-gap 和模型结构图。


工作目录是：<路径> 工作容器是：<容器名>
根据提供的起服务脚步指令，抓取输入4k输出1的trace并进行分析
模型是 <模型名>，框架是 <SGLang/vLLM/TensorRT-LLM>。
请输出一份完整的中文 triage 报告，包含 kernel、overlap、fuse-pattern、bubble-gap 和模型结构图。
```

### 3. `model-architecture-diagram`

作用：查询公开的原始模型架构图，适合 LLM、VLM、MoE、diffusion、OCR 等模型族。

典型能力：

- 返回公开原始图片链接
- 附带来源说明
- 找不到时明确返回 `no_match`

推荐提示词：

```text
$model-architecture-diagram
工作目录是：<路径> 工作容器是：<容器名>
请帮我查询 <模型名> 的公开原始架构图，返回可直接引用的图片 Markdown 和来源说明。
如果没有可用图，请明确告诉我没有匹配结果。
```

### 4. `model-compute-simulation`

作用：做 operator 级的计算模拟，估算 FLOPs、tensor shape、MFU，并用于并行方式和算子路径的 what-if 分析。

典型能力：

- 展开模型的 operator 序列
- 计算关键算子的 FLOPs / tensor shape
- 根据延迟估算 MFU
- 分析 TP / DP / batch / sequence 长度变化的影响

推荐提示词：

```text
$model-compute-simulation
工作目录是：<路径> 工作容器是：<容器名>
请基于模型 <模型名> 和 serving shape <batch/seq/并行配置>，展开 operator 级计算模板，估算关键算子的 FLOPs、tensor shape 和 MFU。
请同时给出不同并行策略或 batch 设定下的对比结论。
```

### 5. `sglang-prod-incident-triage`

作用：SGLang 线上故障排查，采用 replay-first 的流程先还原问题，再决定用 profiler、trace、coredump 还是分布式调试工具。

典型能力：

- 处理 health check 异常
- 排查延迟、吞吐、queue 堆积、timeout、hang
- 分析 crash dump、输出错误、PD/EP/HiCache 问题
- 采集现场 bundle 并回放请求

推荐提示词：

```text
$sglang-prod-incident-triage
工作目录是：<路径> 工作容器是：<容器名>
请帮我排查 SGLang 线上问题。
现象是 <错误现象>，服务地址是 <base url>，我手头有 <日志/请求样本/crash dump/trace>。
请先给出现场采集、回放和下一步 debug 的顺序，不要一上来就做 profiling。
```

### 6. `sglang-sota-humanize-loop`

作用：围绕一个模型自动跑 SGLang SOTA 优化闭环，先做公平 benchmark，再做 profiler 对比，再进入 patch 与 RLCR 迭代。

典型能力：

- 对比 SGLang / vLLM / TensorRT-LLM
- 找出差距并定位根因
- 结合历史知识库和 profiler 证据做 patch
- 迭代到追平或超过最优结果

推荐提示词：

```text
$sglang-sota-humanize-loop
请针对模型 <模型名>，在 <目标环境> 下启动 SGLang SOTA 优化闭环。
请先完成公平 benchmark，再分析领先框架的 profiler 证据，随后给出可执行 patch、验证计划和持续迭代策略。
```

### 7. `vllm-pr-summary`

作用：分析 vLLM 的某个 GitHub PR，输出中文结构化总结、代码分析、技术原理、讨论亮点和风险评估。

典型能力：

- 拉取 PR 数据和 diff
- 生成带 Mermaid 图的报告
- 适合 review、学习和决策前阅读

推荐提示词：

```text
$vllm-pr-summary
请分析 vllm 的 PR #<PR号>，输出一份中文总结报告。
要求覆盖背景、代码修改、技术原理、讨论亮点和风险分析，并给出最终结论。
```

### 8. `model-pr-optimization-history`

作用：查询开源社区 PR 历史，专门为 SGLang / vLLM / TensorRT-LLM 的模型优化提供证据。

典型能力：

- 查找某个模型在社区里做过哪些优化
- 对比 fast path、fusion、overlap、MoE、attention、cache、sampler 等方向
- 为后续 patch 选择提供历史依据

推荐提示词：

```text
$model-pr-history-knowledge
请帮我查询开源社区里与 <框架> / <模型> / <优化方向> 相关的 PR 历史。
重点告诉我之前做过哪些 fast path、fusion、overlap、cache 或 sampler 相关改动，以及它们对当前问题的启发。
```

### 9. `model-pr-optimization-hg-history`

作用：查询本地 `sglang` / `vllm` 工作树中的提交历史，和开源社区 PR 历史互补。

典型能力：

- 查找本地 fork 里的模型优化提交
- 按模型、模块、优化主题检索
- 为 profiling 和 patch 选择提供本地证据

推荐提示词：

```text
$model-pr-history-knowledge
请帮我查询本地仓库里与 <框架> / <模型> / <优化方向> 相关的提交历史。
重点告诉我本地 fork 已经有哪些类似优化、对应文件在哪里、以及我接下来该优先看哪条路径。
```

### 10. `lightop-kernel-knowledge`

作用：面向 LightOp / DCU / ROCm 的 kernel 证据检索 skill，优先查本地实现，再查 ROCm / PyTorch / upstream 证据。

典型能力：

- 找对应 operator 的本地实现与测试
- 查 dispatcher、binding、config table
- 参考 ROCm / HIP / MIOpen / rocBLAS / Triton AMD 证据
- 判断一个 CUDA 思路如何迁移到 DCU / HIP

推荐提示词：

```text
$lightop-kernel-knowledge
请围绕 <算子/模块名> 帮我查 LightOp / DCU / ROCm 的实现证据。
请优先从本地代码和测试出发，再补充 upstream 资料，并给出下一步最值得验证的优化方向。
```

### 11. `hcu-aiter-moe-tuning`

作用：在 HCU ROCm 环境为 AITER ASM fused-MoE 收集、调优、校验和安装配置，适用于新模型、TP/EP shape、FP8/INT8 W8A8 或 gfx936/gfx938 缺少配置的场景。

典型能力：

- 从模型配置、并行策略和运行时权重推导 AITER MoE lookup shape
- 使用 `ASM_MOE_LOG` 收集真实 prefill、decode、并发和分块 token size
- 检查现有 CSV 的精确覆盖，生成 smoke/full 调优输入
- 处理多卡调优、断点续跑、重复行和 runtime key 冲突
- 备份并合并 tuned CSV，验证 AITER lookup、数值精度、性能和 SGLang 服务

推荐提示词：

```text
$hcu-aiter-moe-tuning
模型路径是 <模型路径>，AITER 路径是 <AITER 路径>，目标是 <gfx936/gfx938>，并行配置是 <TP/EP>。
请先从模型和框架代码确认 MoE shape，再收集真实 prefill/decode size、检查已有配置覆盖，并准备可断点续跑的调优包。
未经确认不要占用 GPU、启动服务或修改已安装的 AITER 表。
```

### 12. `hipblaslt-rocblas-tuning`

作用：抓取模型真实 GEMM，对 hipBLASLt 和 NMZ rocBLAS 做 token M 调优、最近兼容点配置补齐，并验证单算子精度与性能，适用于 AMD/海光 GPU 上的 BLAS tuning 和配置交付。

典型能力：

- 以 32768 tokens 输入 / 1 token 输出抓取真实 GEMM，并核对 token 轴（TN 权重左乘时 token M 记作 BLAS `-n`）
- 按真实本地 token M 上限调优 `1,2,4,8,…,cap`，cap 非幂次时补调 cap 本身
- 同一 family 内按绝对距离复制最近调优点，补齐全部整数 M 并保留来源映射
- 分别交付 hipBLASLt override 与 rocBLAS native library 目录，隔离运行副本，不覆盖系统库
- 用 profile/log 的 kernel 命中证据确认实际加载，并校验单算子精度与前后耗时

不覆盖：自定义 MoE / Marlin kernel 调优。

推荐提示词：

```text
$hipblaslt-rocblas-tuning
模型路径是 <模型路径>，目标 GPU 架构是 <gfx 架构>，调优目录是 <隔离目录>。
请先确认 token 轴和 gemm family，再按真实本地 token M 上限调优，补齐全部整数 M，并给出单算子精度与前后耗时对比。
未经确认不要覆盖系统库、占用 GPU 或启动服务。
```

## 安装方式

### 方式一：使用一键安装脚本（推荐）

仓库根目录提供了 `install_all_skills.sh`，用于安装本仓库维护的全部主 skills。默认安装到 Codex 的 skills 目录：

```text
git submodule update --init --recursive
${CODEX_HOME:-$HOME/.codex}/skills
```

基本用法：

```bash
cd /path/to/DCU-Infer-Pilot

git submodule update --init --recursive

# 先预览安装动作，不写入文件
./install_all_skills.sh --dry-run

# 安装到 Codex skills 目录
./install_all_skills.sh

# 已存在同名 skill 时强制覆盖
./install_all_skills.sh --force
```

安装到不同目标：

```bash
# 只安装到 Codex
./install_all_skills.sh --target codex

# 只安装到 Claude
./install_all_skills.sh --target claude

# 同时安装到 Codex 和 Claude
./install_all_skills.sh --target both --force
```

指定安装目录：

```bash
./install_all_skills.sh \
  --target codex \
  --codex-skills-dir /path/to/codex/skills

./install_all_skills.sh \
  --target claude \
  --claude-skills-dir /path/to/claude/skills
```

安装模式：

```bash
# 默认：主 skills 使用软链接安装，Humanize skills 会复制并替换运行时路径
./install_all_skills.sh --link

# 自包含安装：复制主 skills，而不是软链接
./install_all_skills.sh --copy --force
```

脚本会安装本仓库维护的主 skills：

- `llm-serving-auto-benchmark`
- `llm-torch-profiler-analysis`
- `model-architecture-diagram`
- `model-compute-simulation`
- `sglang-prod-incident-triage`
- `sglang-sota-humanize-loop`
- `vllm-pr-summary`
- `model-pr-optimization-history`
- `model-pr-optimization-hg-history`
- `lightop-kernel-knowledge`
- `hcu-aiter-moe-tuning`
- `kb-update`
- `kb-ingest`
- `kb-search`
- `hipblaslt-rocblas-tuning`
- `ask-codex`
- `ask-gemini`
- `humanize`
- `humanize-gen-plan`
- `humanize-refine-plan`
- `humanize-rlcr`
- `lightop-kernel-agent-loop`
- `dcu-profiler-report`

注意：脚本会跳过 cookbook 和 `evidence/pull-bundles` 里嵌套的示例 skills，避免把资料快照中的第三方样例也注册到本地环境。

### 方式二：把单个 skill 安装到 Codex skills 目录

如果你只想用其中几个 skill，最稳妥的方式是把对应目录复制到 Codex 的 skills 根目录：

```bash
mkdir -p ~/.codex/skills

cp -a /path/to/DCU-Infer-Pilot/skills/llm-serving-auto-benchmark ~/.codex/skills/
cp -a /path/to/DCU-Infer-Pilot/skills/llm-torch-profiler-analysis ~/.codex/skills/
cp -a /path/to/DCU-Infer-Pilot/skills/model-architecture-diagram ~/.codex/skills/
cp -a /path/to/DCU-Infer-Pilot/skills/hcu-aiter-moe-tuning ~/.codex/skills/
cp -a /path/to/DCU-Infer-Pilot/skills/hipblaslt-rocblas-tuning ~/.codex/skills/
```

对知识库型内容，也可以按需一并复制：

```bash
cp -a /path/to/DCU-Infer-Pilot/model-pr-optimization-history ~/.codex/skills/
cp -a /path/to/DCU-Infer-Pilot/model-pr-optimization-hg-history ~/.codex/skills/
```

### 方式三：把整个仓库放到可扫描位置

如果你的 Codex 环境支持递归扫描子目录，可以直接把整个仓库放到 skills 根目录下管理：

```bash
mkdir -p ~/.codex/skills
cp -a /path/to/DCU-Infer-Pilot ~/.codex/skills/
```

### 方式四：从 git 安装

```bash
git clone <repo-url> DCU-Infer-Pilot
cd DCU-Infer-Pilot
./install_all_skills.sh
```

## 使用方式

安装后，在对话里直接用技能名触发：

```text
$llm-serving-auto-benchmark ...
$llm-torch-profiler-analysis ...
$sglang-prod-incident-triage ...
$hcu-aiter-moe-tuning ...
$hipblaslt-rocblas-tuning ...
```

一般建议你一次只触发一个主 skill，再让它调用需要的历史知识库或辅助技能。比如：

- 做 benchmark 时，优先用 `llm-serving-auto-benchmark`
- 看 trace 时，优先用 `llm-torch-profiler-analysis`
- 排查线上问题时，优先用 `sglang-prod-incident-triage`
- 做 SGLang 持续优化时，优先用 `sglang-sota-humanize-loop`
- 查历史证据时，优先用 `model-pr-optimization-history` 或 `model-pr-optimization-hg-history`
- 调 GEMM 算子库时，优先用 `hipblaslt-rocblas-tuning`

## 测试与脚本

仓库里还带有回归测试和重建脚本，用来保证 skill 文档、查询逻辑和结果格式稳定：

- `tests/`：基础回归测试
- `tools/rebuild_model_pr_history_from_git.py`：重建社区 PR 历史
- `tools/rebuild_model_pr_hg_history_from_local_git.py`：重建本地提交历史

运行测试：

```bash
cd DCU-Infer-Pilot
python3 -m pytest tests
```

如果默认 `python3` 版本过老，改用系统里的新版本即可：

```bash
/usr/bin/python3.8 -m pytest tests
/usr/bin/python3.10 -m pytest tests
```

## 使用建议

- 做性能优化时，先 benchmark，再 profiler，再查历史证据，最后 patch。
- 处理线上异常时，先采集现场和可复现请求，再决定要不要 profiling。
- 如果目标是 LightOp / DCU / ROCm kernel 优化，先用知识 skill 找证据，再用 profiling skill 量化瓶颈。
- `ask-codex` 和 `ask-gemini` 适合做辅助研究，不适合作为主流程替代品。
