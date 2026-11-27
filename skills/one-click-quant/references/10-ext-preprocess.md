# 扩展：新模型预处理接入

> **权威来源**：`docs/preprocess.md`（接入流程）、`docs/preprocess-theory.md`（QuaRot / Flex Smooth / AWQ-SSZ 数学原理）。
> 本文只写**接入必须做什么**与**怎么验证接入是对的**；预处理产物是浮点权重，与下游量化**解耦**。

## 架构分层（不要重新发明）

```text
WeightStore / pipeline / absorb / transforms   ← 通用层，复用
adapters/<model>.py                            ← 每模型一份
recipes/<model>.yaml                           ← 每模型一份
```

## Adapter 钩子（`preprocess/adapters/base.py`）

| 方法 | 必需 | 作用 |
|---|---|---|
| `build_ln_fuse_map` | **抽象方法** | `norm_module -> [linear, ...]`（不带 `.weight`）。同时是 Flex Smooth 「一个 Norm 必须和它的全部 consumer 一起缩放」的判定依据 |
| `build_rotate_pairs` | **抽象方法** | 返回 `([(left_map, right_map), ...], rot_meta)`；不用 QuaRot 时返回 `[], {}` |
| `resolve_up_down` | 默认 `None` | UpDown 种子（`...up_proj` / `...w3`）→ down partner；返回 `None` 表示该模型不适用 |
| `resolve_norm_linear` | 默认查 `build_ln_fuse_map` | Flex Smooth 种子的 `(norm, [consumers])`；仅当 Smooth 分组与 LN-fuse 分组不同时才覆盖 |
| `norm_affine` | 默认 `"rms"` | **`"rms"`（gamma 初始化为 1）或 `"gemma"`（`1+gamma`）**。用错会让 Smooth 的 `gamma /= s` 数学算错，必须按模型 RMSNorm 的实际实现选 |
| `hidden_smooth_partners` | 默认 `[]` | 共享同一 hidden 维 Smooth scale 的额外 Linear（如 Gemma 里 wide 与 mean-mixed 两类 consumer） |
| `prepare_weights` | 默认 no-op | transform 之前改权重的钩子（如 block FP8 / MXFP4 就地解包），`ctx` 带 pipeline 上下文 |

> `norm_affine` / `hidden_smooth_partners` 易漏。`norm_affine` 按该模型 RMSNorm 实现选
> `"rms"`（gamma≈1）或 `"gemma"`（`1+gamma`）；**以该模型 checkpoint / 已有 adapter 为准，勿按族名抄**。

### 各 transform 需要的钩子

| Transform | 需要 adapter 提供 | recipe 中的种子 |
|-----------|-------------------|-----------------|
| `quarot` | `build_ln_fuse_map`、`build_rotate_pairs` | 可选 `quarot.*`；在 `transforms` 中列出 |
| `flex_awq_ssz` | `resolve_up_down` | `flex_awq_ssz.include`（up / gate_up 种子） |
| `flex_smooth_quant` | `build_ln_fuse_map` / `resolve_norm_linear` | `flex_smooth_quant.include`（norm 种子） |

`prepare_weights` 以代码为准：`grep -n "def prepare_weights" -A5 preprocess/adapters/*.py`。
默认 block FP8 解包走 staging：recipe `unpack_fp8`（默认 `true`）→ `{save-dir}.bf16_staging`。
文档/注释里点名的 adapter 若目录中不存在，以 `ls` 为准。

已注册的 transform：`quarot`、`flex_smooth_quant`、`flex_awq_ssz`（见 `preprocess/transforms/__init__.py` 的 `TRANSFORM_REGISTRY`）。

## 各 transform 可开条件（硬门槛）

> **可开当且仅当下列条件同时满足**；任一不满足 → 跳过该 transform，禁止在不满足条件时强制启用。
> 选用顺序服从 `00` 阶段 F（Dense 精度升级优先 QuaRot；MoE 按条件自适应）。

### QuaRot（`quarot`）

须**同时**满足：

1. **残差与 hidden 维一致**：普通 `x+attn` / `x+mlp`，或可对残差做 block-diag 重复同一旋转 R。
2. **Norm 可 fuse 的 RMSNorm**：fuse 后 Norm 输出**不得**再被非 Linear 路径复用（典型反例：`gates * normed`）。
3. adapter 有完整左右旋成对表：`build_rotate_pairs` 返回非空、可配对的 left/right map。
4. **没有**「对角缩放挂在残差上且与 Hadamard 不对易」的结构（典型反例：Hyper-Connection + 未 fuse 的 Gemma 式 `hc_norm`）。

### AWQ（`flex_awq_ssz`）

须**同时**满足：

1. 有成对 **Up–Down**，中间维一致（`up.out == down.in`，或 fused `gate_up` 半宽对齐）；adapter 实现可用的 `resolve_up_down`。
2. 中间维对角缩放可离线吸收（中间允许 SiLU / SwiGLU）。
3. recipe 给出明确 **up 种子** `include`（一般只打最尖的 up，如 `up_proj` / `w3`）。
4. 权重已是**浮点**（未量化；block FP8/MXFP4 须先解包到浮点）。

### SmoothQuant（`flex_smooth_quant`）

须**同时**满足：

1. Norm→Linear 消费者表可**完整枚举**（`build_ln_fuse_map` / `resolve_norm_linear`）；漏兄弟 Linear 会破坏等价。
2. 缩放轴 = Norm 输出维 = 各 Linear 的 `in_features`。
3. recipe `include` 种子能解析进某 Norm 组；**解析失败则跳过，禁止静默改单边**。
4. 权重与 γ 已是**浮点**。

**adapter 有哪些，以 `ADAPTER_REGISTRY` 的实际内容为准**，不要按本文列举（文档写"已接入 N 个"一定会过期）：

```bash
grep -n "ADAPTER_REGISTRY" -A20 preprocess/adapters/__init__.py      # 不依赖 torch，总能跑
# 环境就绪时也可以用运行时自省：
python3 -c "from preprocess.adapters import ADAPTER_REGISTRY as R; print(sorted(R))"
```

> 注意 `preprocess/adapters/__init__.py` 里有**两处过期提示**：`get_adapter` / `resolve_adapter_name`
> 的报错文案仍把 `kimi_k3` 当成合法值举例，且 `resolve_adapter_name` 对 kimi 模型仍会返回 `"kimi_k3"`。
> 这两处都不代表该 adapter 存在 —— 判据只有 `ADAPTER_REGISTRY` 里有没有这个 key。

## 检查清单

复制后逐项勾选：

```text
- [ ] adapters/<model>.py 实现 ModelAdapter
- [ ] `norm_affine` 与 `hidden_smooth_partners` 已按该模型的实际 Norm 语义确认（默认值是否够用）
- [ ] 在 preprocess/adapters/__init__.py 的 ADAPTER_REGISTRY 注册（必要时加别名）
- [ ] recipes/<model>.yaml 写明 adapter: <name> 以及 transforms/include
- [ ] python3 ./skills/scripts/check_recipe.py preprocess/recipes/<model>.yaml
- [ ] 冒烟模型用 build_mini_*.py 做全覆盖减层（见下节，禁止随手切几层）
- [ ] 冒烟：在小模型上跑通 recipe 并写出浮点 HF 目录（`quarot.max_layers` 只限制**旋转/融合**的层数，
      不减少产物层数，别指望它代替减层）
- [ ] 下游：model_free_ptq / 通用 PTQ 加 --from-preprocessed 能加载
```

## 冒烟模型：必须全覆盖减层

小模型的用途是**证明 adapter 覆盖了该模型实际拥有的每一种结构单元**，不是把显存压小。

> 减层 ≠ 取前 N 层。候选结构见 `02-model-prep.md`「结构覆盖要求」：**有则覆盖，无则跳过**。
> 对含 MTP / linear_attn / 晚出现 full-attn 的模型，只取前 4 层常漏覆盖，冒烟会假通过。

做法：

1. 有 `--dry-run` 则先扫不写盘（以该脚本 `--help` 为准）
2. 层号不连续才能覆盖多样结构时，用 `--layer-ids` 显式指定
3. `-n` / `-e` 默认值因脚本而异；冒烟时可减小 expert 数

```bash
# 先干跑看筛选
python3 examples/<model>/build_mini_<model>.py \
  --src <真实源模型目录> --dst <真实输出目录> -n 4 -e 8 --dry-run

# 再实跑
python3 examples/<model>/build_mini_<model>.py \
  --src <真实源模型目录> --dst <真实输出目录> -n 4 -e 8
```

选用哪个脚本、各脚本 flag 差异、结构覆盖清单，见 `02-model-prep.md`。

## Adapter 骨架

```python
# preprocess/adapters/my_model.py
from preprocess.adapters.base import ModelAdapter

class MyModelAdapter(ModelAdapter):
    name = "my_model"

    def build_ln_fuse_map(self, cfg, num_layers=None):
        ...

    def build_rotate_pairs(self, cfg, *, block_size=32, num_layers=None, num_experts=None):
        # 不用 QuaRot 时返回 [], {}
        return [], {}

    def resolve_up_down(self, up_module: str):
        # 例如 "*.up_proj" -> "*.down_proj"
        ...

    # gemma 系 Norm 必须覆盖；rms 系保持默认即可
    def norm_affine(self) -> str:
        return "rms"
```

注册（二选一，`register_adapter` 会顺带校验 `name` 非空）：

```python
# preprocess/adapters/__init__.py
from preprocess.adapters.my_model import MyModelAdapter

ADAPTER_REGISTRY[MyModelAdapter.name] = MyModelAdapter
# 或（注意 register_adapter 定义在本文件靠后，调用要放在它之后）
register_adapter(MyModelAdapter)
```

## Recipe 骨架

```yaml
adapter: my_model
transforms:
  - flex_awq_ssz
  - flex_smooth_quant

flex_awq_ssz:
  search_step: 0.1
  ssz_step: 10
  use_calib_acts: false
  include:
    - model.layers.*.mlp.up_proj

flex_smooth_quant:
  search_step: 0.1
  use_calib_acts: false
  include:
    - model.layers.*.input_layernorm
    - model.layers.*.post_attention_layernorm
```

已有 recipe：以 `ls preprocess/recipes/` 为准，选同结构最近的一份再改。

## 运行

```bash
python3 main.py --alg preprocess \
  --model <真实模型目录> \
  --save-dir <真实输出目录> \
  --recipe preprocess/recipes/my_model.yaml \
  --transforms flex_awq_ssz,flex_smooth_quant \
  --preprocess-gpus 8
```

- `--preprocess-gpus N`：按层数据并行改权重（不是 MoE EP）。单卡用默认 `1`，可选 `--preprocess-device cuda:0`。
- CLI 的 `--transforms` 会覆盖 recipe 中的顺序。
- **必须显式传 `--recipe`**：`--recipe` 省略时会落到 `preprocess/pipeline.py::_default_recipe_path()`
  给出的路径，而那个文件不在仓库里（`main.py --recipe` 的 help 文案和 `templates/preprocess_ptq.py`
  的 docstring 都还是这个过期路径 —— 文案不可信，以 `ls preprocess/recipes/` 为准）。
  不传 `--recipe` 会直接 `FileNotFoundError`。
- 真实激活校准：先采集 `acts.pt`，再传 `--act-stats`，并在 recipe 的**对应 transform 块下**
  设 `use_calib_acts: true`（顶层写这个键不生效；见 `docs/preprocess.md`）。

## 验收标准（全部满足才算接入成功）

- 产物是浮点权重，且 `config.json` 里**两个量化键都不存在**：`compression_config` 与 `quantization_config`
  （`assert_float_checkpoint` 检查的是这两个键，且**顶层与 `text_config` 内都要查**；只看其中一个会漏）
- 冒烟 mini 覆盖了该模型**每一种结构单元**（对照 `02-model-prep.md` 的结构覆盖清单逐项确认）
- 量化路径能接受该目录（加 `--from-preprocessed`），且产物通过 `04-verify-eval.md` 的契约校验
- 新模型接入没有改动核心 `absorb` / `pipeline` / transform 代码
