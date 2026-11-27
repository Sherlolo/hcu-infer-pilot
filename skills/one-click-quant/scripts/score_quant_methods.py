#!/usr/bin/env python3
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""按匹配度规则对候选量化方法打分并排序。

匹配度 = 工程可行性 + 结构/权重/目标契合（0–100），**不是** ΔNLL 预测。
分值唯一来源：同目录 score_rubric.yaml（可用 --rubric 覆盖）。
开发说明：skills/references/07-match-score-rubric.md
用法见 06-method-model-map.md。

用法：
  python3 ./skills/scripts/score_quant_methods.py --model /path/to/HF_model
  python3 ./skills/scripts/score_quant_methods.py --model /path/to/HF_model \\
      --target int8 --size-gb 140 --format markdown

退出码：0 成功；2 用法/输入非法。
兼容 Python 3.6+。
"""

from __future__ import print_function

import argparse
import json
import os
import re
import subprocess
import sys

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
EXAMPLES_DIR = os.path.join(REPO_ROOT, "examples")
YAML_PATH = os.path.join(REPO_ROOT, "templates", "quant_templates.yaml")
MOE_RUN_DIR = os.path.join(REPO_ROOT, "utils", "MoE-Quant", "tests")
DEFAULT_RUBRIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "score_rubric.yaml")

# Loaded once per process via load_rubric(); scoring reads this.
RUBRIC = None

_FAMILY_HINTS = {
    "deepseek": ("deepseek",),
    "glm": ("glm",),
    "qwen": ("qwen",),
    "kimi": ("kimi",),
    "mimo": ("mimo",),
    "minimax": ("minimax",),
    "hy4": ("hy4", "hy_"),
    "step": ("step",),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Score quantization methods for a HF model (06 match rubric)."
    )
    p.add_argument("--model", required=True, help="Local HF model directory")
    p.add_argument(
        "--target",
        default="auto",
        choices=["auto", "fp8", "int8", "w4", "w4a8"],
        help="Customer target format (default auto)",
    )
    p.add_argument(
        "--size-gb",
        type=float,
        default=None,
        help="Override model dir size in GB (skip du)",
    )
    p.add_argument(
        "--weight",
        default=None,
        choices=[
            "bf16_fp16",
            "block_fp8",
            "w4",
            "mxfp4",
            "already_quantized",
            "unknown",
        ],
        help="Override weight form tag",
    )
    p.add_argument(
        "--has-history",
        action="store_true",
        help="Mark as adapted (has quant history)",
    )
    p.add_argument(
        "--format",
        default="markdown",
        choices=["markdown", "json"],
        help="Output format",
    )
    p.add_argument(
        "--min-score",
        type=int,
        default=1,
        help="Hide candidates with score below this",
    )
    p.add_argument("--top", type=int, default=0, help="Only print top N (0=all)")
    p.add_argument(
        "--rubric",
        default=DEFAULT_RUBRIC,
        help="Path to score_rubric.yaml (default: alongside this script)",
    )
    return p.parse_args(argv)


def load_rubric(path):
    """Load score_rubric.yaml; raises IOError/ValueError on failure."""
    if not os.path.isfile(path):
        raise IOError("rubric not found: %s" % path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError("rubric root must be a mapping: %s" % path)
    for key in ("score_clamp", "gates", "llmc_base", "factors", "target_bonus"):
        if key not in data:
            raise ValueError("rubric missing required key %r: %s" % (key, path))
    return data


def _load_config(model):
    cfg_path = os.path.join(model, "config.json")
    if not os.path.isfile(cfg_path):
        raise IOError("missing config.json under %s" % model)
    with open(cfg_path, "r") as f:
        return json.load(f)


def _estimate_size_gb(model):
    try:
        out = subprocess.check_output(["du", "-sb", model], stderr=subprocess.DEVNULL)
        if not isinstance(out, str):
            out = out.decode("utf-8", "replace")
        nbytes = int(out.split()[0])
        return round(nbytes / float(1024 ** 3), 2)
    except Exception:
        return None


def _infer_weight(model, cfg):
    qcfg = cfg.get("quantization_config") or {}
    if qcfg:
        fmt = str(qcfg.get("quant_method") or qcfg.get("format") or "").lower()
        if "fp8" in fmt or "float8" in fmt:
            return "block_fp8"
        if "mxfp4" in fmt or "nvfp4" in fmt:
            return "mxfp4"
        return "already_quantized"
    keys = []
    index = os.path.join(model, "model.safetensors.index.json")
    if os.path.isfile(index):
        try:
            with open(index, "r") as f:
                keys = list(json.load(f).get("weight_map", {}).keys())[:200]
        except Exception:
            keys = []
    blob = " ".join(keys).lower()
    if "weight_scale_inv" in blob:
        return "block_fp8"
    for k in keys:
        if "weight_packed" in k or ".qweight" in k:
            return "w4"
    try:
        for name in os.listdir(model):
            if name.endswith(".safetensors"):
                low = name.lower()
                if "fp8" in low:
                    return "block_fp8"
                if "mxfp4" in low:
                    return "mxfp4"
                if "int4" in low:
                    return "w4"
                break
    except OSError:
        pass
    return "bf16_fp16"


def _infer_topology(cfg, mt, arch_l):
    text = cfg.get("text_config") or {}
    clues = [
        cfg.get("num_local_experts"),
        cfg.get("n_routed_experts"),
        cfg.get("num_experts"),
    ]
    if isinstance(text, dict):
        clues.extend(
            [text.get("num_local_experts"), text.get("n_routed_experts")]
        )
    if any(clues):
        return "moe"
    joined = " ".join([mt] + list(arch_l))
    for x in ("moe", "mixtral", "deepseek_v3", "glm_moe", "kimi_k3"):
        if x in joined:
            return "moe"
    return "dense"


def _infer_vl(cfg, mt, arch_l):
    joined = " ".join([mt] + list(arch_l)).lower()
    if "vl" in mt or "qwen2_5_vl" in joined or "qwen2_vl" in joined:
        return True
    if "vision_config" in cfg or "visual" in cfg:
        return True
    arch_s = "".join(arch_l)
    if "forconditionalgeneration" in arch_s and "moe" not in joined:
        if "vl" in joined or "vision" in joined:
            return True
    return False


def _scan_examples(mt, arch_l, model_name):
    """Return list of (rel_path, strength) where strength is 'strong'|'weak'.

    strong: script stem shares meaningful tokens with model/arch.
    weak: only family-name hint (e.g. both under qwen/) — lower score later.
    """
    hits = []
    hay = " ".join([mt, model_name.lower()] + list(arch_l)).lower()
    if not os.path.isdir(EXAMPLES_DIR):
        return hits
    tokens = [t for t in re.split(r"[^a-z0-9]+", hay) if len(t) >= 4]
    # drop ultra-generic tokens that cause false family hits
    skip = set(
        [
            "qwen",
            "llama",
            "model",
            "forcausallm",
            "forconditionalgeneration",
            "instruct",
            "chat",
        ]
    )
    tokens = [t for t in tokens if t not in skip]
    for family, hints in _FAMILY_HINTS.items():
        if not any(h in hay for h in hints):
            continue
        fam_dir = os.path.join(EXAMPLES_DIR, family)
        if not os.path.isdir(fam_dir):
            continue
        for name in sorted(os.listdir(fam_dir)):
            if not name.endswith(".py"):
                continue
            if name.startswith("build_mini") or name.startswith("compare_"):
                continue
            stem = name[:-3].lower()
            overlap = sum(1 for t in tokens if t in stem)
            rel = "examples/%s/%s" % (family, name)
            if overlap >= 1:
                hits.append((rel, "strong"))
            elif family in hay:
                hits.append((rel, "weak"))
    seen = set()
    out = []
    for h in hits:
        if h[0] not in seen:
            seen.add(h[0])
            out.append(h)
    return out[:12]


def _moe_adapter(mt, arch_l):
    arch_s = "".join(arch_l)
    if mt == "mimo_v2":
        return "MiMoV2Adapter"
    if mt == "step5_vl":
        return "Step5Adapter"
    if mt == "kimi_k3" or "kimik3forconditionalgeneration" in arch_s:
        return "KimiK3Adapter"
    if (
        mt == "glm5_next"
        or "glm5nextforconditionalgeneration" in arch_s
        or "glm5nextforcausallm" in arch_s
    ):
        return "Glm5NextAdapter"
    if mt == "glm_moe_dsa" or "glmmoedsaforcausallm" in arch_s:
        return "GlmMoeDsaAdapter"
    if mt in ("qwen3_5_moe", "qwen3_5_moe_text") or "qwen3_5moe" in arch_s:
        return "Qwen35MoeAdapter"
    if mt == "deepseek_v3" or "deepseekv3forcausallm" in arch_s:
        return "DeepSeekV3Adapter"
    return None


def _yaml_schemes_for_arch(architectures):
    if not os.path.isfile(YAML_PATH):
        return ["FP8_TO_BF16"]
    with open(YAML_PATH, "r") as f:
        text = f.read()
    found = []
    for arch in architectures:
        if not re.search(r"^%s:\s*$" % re.escape(arch), text, re.M):
            continue
        m = re.search(
            r"^%s:\s*\n(.*?)(?=^[A-Za-z\"*]|\Z)" % re.escape(arch),
            text,
            re.M | re.S,
        )
        block = m.group(1) if m else ""
        for sm in re.finditer(r"^\s{6}([A-Za-z0-9_]+):\s*$", block, re.M):
            found.append(sm.group(1))
    if "FP8_TO_BF16" not in found:
        found.append("FP8_TO_BF16")
    return found


def build_profile(args):
    model = os.path.abspath(os.path.expanduser(args.model))
    if not os.path.isdir(model):
        raise IOError("model dir not found: %s" % model)
    cfg = _load_config(model)
    mt = str(cfg.get("model_type") or "").lower()
    architectures = [str(a) for a in (cfg.get("architectures") or [])]
    arch_l = set(a.lower() for a in architectures)
    if args.size_gb is not None:
        size_gb = args.size_gb
    else:
        size_gb = _estimate_size_gb(model)
    weight = args.weight or _infer_weight(model, cfg)
    topology = _infer_topology(cfg, mt, arch_l)
    is_vl = _infer_vl(cfg, mt, arch_l)
    examples_hits = _scan_examples(mt, arch_l, os.path.basename(model))
    moe = _moe_adapter(mt, arch_l)
    yaml_schemes = _yaml_schemes_for_arch(architectures)
    notes = []
    if size_gb is None:
        notes.append("size_gb unknown (pass --size-gb for accurate gate)")
    return {
        "model": model,
        "model_type": mt,
        "architectures": architectures,
        "arch_l": arch_l,
        "size_gb": size_gb,
        "topology": topology,
        "is_vl": is_vl,
        "weight": weight,
        "target": args.target,
        "has_quant_history": bool(args.has_history),
        "examples_hits": examples_hits,
        "moe_adapter": moe,
        "yaml_schemes": yaml_schemes,
        "notes": notes,
    }


def _clamp(n):
    lo = int(RUBRIC["score_clamp"]["min"])
    hi = int(RUBRIC["score_clamp"]["max"])
    return max(lo, min(hi, n))


def _F(*keys):
    """Fetch nested int from RUBRIC['factors'] ..."""
    node = RUBRIC["factors"]
    for k in keys:
        node = node[k]
    return int(node)


def _B(name):
    return int(RUBRIC["llmc_base"][name])


def _target_bonus(target, kind):
    tb = RUBRIC["target_bonus"]
    if target == "auto":
        return int(tb["auto_neutral"])
    mapping = tb.get(target) or {}
    if kind not in mapping:
        return 0
    return int(mapping[kind])


def _add(bd, key, val, pros, cons, pro="", con=""):
    bd[key] = bd.get(key, 0) + val
    if val > 0 and pro:
        pros.append(pro)
    if val < 0 and con:
        cons.append(con)


def _cand(method_id, path, score, pros, cons, breakdown, command_hint,
          excluded=False, exclude_reason=""):
    return {
        "method_id": method_id,
        "path": path,
        "score": 0 if excluded else score,
        "pros": pros,
        "cons": cons,
        "breakdown": breakdown,
        "command_hint": command_hint,
        "excluded": excluded,
        "exclude_reason": exclude_reason,
    }


def score_candidates(p):
    cands = []
    gate_gb = float(RUBRIC["gates"]["dense_ultra_gb"])
    ultra = p["size_gb"] is not None and p["size_gb"] > gate_gb
    dense_ultra_block = ultra and p["topology"] == "dense"

    def make_llmc(method_id, scheme, alg, kind, base, family_sch=None):
        bd = {}
        pros = []
        cons = []
        hint = "python3 main.py --model <MODEL> --scheme %s" % scheme
        if alg:
            hint += " --alg %s" % alg
        hint += " --save-dir <SAVE>"
        if dense_ultra_block:
            return _cand(
                method_id,
                "llmc",
                0,
                [],
                [
                    "未适配 Dense >%sGB 禁止 llmc（00 超大规则）"
                    % int(gate_gb)
                ],
                {"hard_gate": 0},
                hint,
                excluded=True,
                exclude_reason="dense_ultra_llmc_forbidden",
            )
        score = base
        bd["base"] = base
        if p["topology"] == "dense":
            v = _F("topology", "dense_llmc")
            _add(bd, "topology", v, pros, cons, pro="Dense → llmc 默认路径契合")
            score += v
        else:
            v = _F("topology", "moe_llmc")
            _add(
                bd,
                "topology",
                v,
                pros,
                cons,
                con="MoE 上通用 llmc 往往不如 examples/MoE-Quant",
            )
            score += v
        if p["weight"] == "bf16_fp16" and kind in ("fp8", "int8", "w4", "w4a8"):
            v = _F("weight", "bf16_generic_match")
            _add(
                bd,
                "weight",
                v,
                pros,
                cons,
                pro="源为 BF16/FP16，与该 scheme 输入匹配",
            )
            score += v
        elif p["weight"] == "block_fp8" and kind == "unpack":
            v = _F("weight", "block_fp8_unpack_match")
            _add(bd, "weight", v, pros, cons, pro="源为 block FP8，解包路径匹配")
            score += v
        elif p["weight"] == "block_fp8" and kind != "unpack":
            v = _F("weight", "block_fp8_generic_mismatch")
            _add(
                bd,
                "weight",
                v,
                pros,
                cons,
                con="源已是 block FP8，不宜当 BF16 直接套该 scheme",
            )
            score += v
        elif p["weight"] == "w4" and kind in ("int8", "fp8"):
            v = _F("weight", "w4_as_bf16_scheme")
            _add(
                bd,
                "weight",
                v,
                pros,
                cons,
                con="源已是 W4，通用 BF16 scheme 可能不适用",
            )
            score += v
        if p["size_gb"] is not None and p["size_gb"] <= gate_gb:
            v = _F("size", "llmc_within_gate")
            _add(
                bd,
                "size",
                v,
                pros,
                cons,
                pro="规模约 %sGB ≤%sGB，可用 llmc"
                % (p["size_gb"], int(gate_gb)),
            )
            score += v
        smooth_default = scheme in ("W8A8", "W4A8", "W4AFP8") and alg is None
        if p["is_vl"] and smooth_default:
            v = _F("vl", "smooth_default_penalty")
            _add(
                bd,
                "vl_mapping",
                v,
                pros,
                cons,
                con="VL/嵌套层上默认 SmoothQuant mapping 易失败",
            )
            score += v
        if p["is_vl"] and alg == "gptq":
            v = _F("vl", "gptq_bonus")
            _add(
                bd,
                "vl_mapping",
                v,
                pros,
                cons,
                pro="显式 --alg gptq，避开 SmoothQuant 默认 mapping",
            )
            score += v
        if p["is_vl"] and alg == "ptq":
            v = _F("vl", "ptq_bonus")
            _add(bd, "vl_mapping", v, pros, cons, pro="ptq 无 SmoothQuant，VL 更稳")
            score += v
        if p["is_vl"] and alg == "model_free_ptq":
            v = _F("vl", "model_free_bonus")
            _add(
                bd,
                "vl_mapping",
                v,
                pros,
                cons,
                pro="model_free_ptq 不走 HF Smooth 路径，VL 可试",
            )
            score += v
            cons.append("须相对路径软链（03 §3）")
        if p["has_quant_history"]:
            v = _F("history", "llmc_side_evidence")
            _add(
                bd,
                "history",
                v,
                pros,
                cons,
                con="已有量化历史时应以历史菜单为准，本候选仅作辅助对照",
            )
            score += v
        tb = _target_bonus(p["target"], kind)
        _add(
            bd,
            "target",
            tb,
            pros,
            cons,
            pro=("与客户目标 %s 对齐 (+%s)" % (p["target"], tb)) if tb > 0 else "",
            con=(
                ("与客户目标 %s 不完全对齐 (%s)" % (p["target"], tb))
                if tb < 0
                else ""
            ),
        )
        score += tb
        if scheme == "FP8_DYNAMIC":
            pros.append("免校准或弱校准；Dense 匹配度常居前列")
            cons.append("压缩比通常弱于 W4")
        if alg == "gptq":
            cons.append("GPTQ 校准耗时/显存较高")
        if alg == "model_free_ptq":
            pros.append("免校准、通常更快")
            cons.append("精度不保证优于 GPTQ")
        if smooth_default:
            cons.append("默认需校准数据，墙钟较长")
        if family_sch:
            fam_pen = _F("family", "weight_mismatch")
            if "DeepseekV3_FP8" in family_sch and p["weight"] != "block_fp8":
                _add(
                    bd,
                    "family_weight",
                    fam_pen,
                    pros,
                    cons,
                    con="族模板要 block FP8 源",
                )
                score += fam_pen
            if "Kimi_W4" in family_sch and p["weight"] not in (
                "w4",
                "mxfp4",
                "already_quantized",
            ):
                _add(
                    bd,
                    "family_weight",
                    fam_pen,
                    pros,
                    cons,
                    con="族模板要 W4 源",
                )
                score += fam_pen
            v = _F("family", "yaml_hit")
            _add(
                bd,
                "yaml_hit",
                v,
                pros,
                cons,
                pro="quant_templates 命中 scheme %s" % family_sch,
            )
            score += v
        if method_id.startswith("llmc:W4A8"):
            cons.append("≠ MoE-Quant「仅路由专家/混合 W4A8」档位")
        return _cand(
            method_id, "llmc", _clamp(score), pros, cons, bd, hint
        )

    cands.append(
        make_llmc(
            "llmc:FP8_DYNAMIC", "FP8_DYNAMIC", None, "fp8", _B("FP8_DYNAMIC")
        )
    )
    cands.append(
        make_llmc(
            "llmc:W8A8(default smoothquant_gptq)",
            "W8A8",
            None,
            "int8",
            _B("W8A8_default"),
        )
    )
    cands.append(
        make_llmc("llmc:W8A8+gptq", "W8A8", "gptq", "int8", _B("W8A8_gptq"))
    )
    cands.append(
        make_llmc("llmc:W8A8+ptq", "W8A8", "ptq", "int8", _B("W8A8_ptq"))
    )
    cands.append(
        make_llmc(
            "llmc:W8A8+model_free_ptq",
            "W8A8",
            "model_free_ptq",
            "int8",
            _B("W8A8_model_free_ptq"),
        )
    )
    cands.append(make_llmc("llmc:W4A16", "W4A16", None, "w4", _B("W4A16")))
    cands.append(
        make_llmc("llmc:W4A16_ASYM", "W4A16_ASYM", None, "w4", _B("W4A16_ASYM"))
    )
    cands.append(
        make_llmc("llmc:W4A8(llmc配方名)", "W4A8", None, "w4a8", _B("W4A8"))
    )
    cands.append(
        make_llmc(
            "llmc:FP8_TO_BF16", "FP8_TO_BF16", None, "unpack", _B("FP8_TO_BF16")
        )
    )

    for sch in p["yaml_schemes"]:
        if sch in ("FP8_DYNAMIC", "W8A8", "FP8_TO_BF16"):
            continue
        kind = "int8"
        if "FP8" in sch and "INT8" not in sch:
            kind = "fp8"
        cands.append(
            make_llmc(
                "llmc:%s" % sch,
                sch,
                None,
                kind,
                _B("family_scheme"),
                family_sch=sch,
            )
        )

    for item in p["examples_hits"]:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            hit, strength = item[0], item[1]
        else:
            hit, strength = item, "strong"
        if strength == "strong":
            b0 = _F("examples", "strong_base")
            b1 = _F("examples", "strong_hit")
            bd = {"base": b0, "examples_hit": b1}
            score = b0 + b1
            pros = ["仓库已有近亲脚本 %s" % hit, "MoE/Flash/混合精度常见首选"]
            cons = [
                "非统一 main.py；须读该目录 README",
                "脚本参数可能绑定特定权重形态",
            ]
        else:
            b0 = _F("examples", "weak_base")
            b1 = _F("examples", "weak_family_only")
            bd = {"base": b0, "examples_family_only": b1}
            score = b0 + b1
            pros = ["同 family 目录下有脚本 %s（仅族名相近）" % hit]
            cons = [
                "非同结构/同版本强匹配，勿直接当历史配方",
                "须读 README 确认是否适用于本模型",
            ]
        if p["topology"] == "moe" and strength == "strong":
            v = _F("topology", "moe_examples_strong")
            bd["topology"] = v
            score += v
            pros.append("MoE → examples 优先级高于通用 llmc")
        elif p["topology"] == "dense" and strength == "strong":
            v = _F("topology", "dense_examples_strong")
            bd["topology"] = v
            score += v
        tb = _target_bonus(p["target"], "mixed")
        bd["target"] = tb
        score += tb
        if p["has_quant_history"] and strength == "strong":
            pros.append("若历史即该脚本，匹配度应视为最高档")
            score = min(
                int(RUBRIC["score_clamp"]["max"]),
                score + _F("examples", "history_boost"),
            )
        cands.append(
            _cand(
                "examples:%s" % hit,
                "examples",
                _clamp(score),
                pros,
                cons,
                bd,
                "python3 %s  # 参数见同目录 README" % hit,
            )
        )

    strong_examples = [
        x[0] if isinstance(x, (list, tuple)) else x
        for x in p["examples_hits"]
        if (isinstance(x, (list, tuple)) and x[1] == "strong")
        or not isinstance(x, (list, tuple))
    ]
    if p["topology"] == "moe" and not strong_examples:
        aq_base = _F("autoquantize", "base")
        aq_bonus = (
            _F("autoquantize", "moe_adapter_bonus") if p["moe_adapter"] else 0
        )
        score = aq_base + aq_bonus
        cands.append(
            _cand(
                "autoquantize→examples脚本落地",
                "autoquantize",
                _clamp(score),
                [
                    "无现成 examples 时的标准升级（00 F2）",
                    "可产出混合位宽 plan → scheme",
                ],
                [
                    "AutoQuantize 本身耗时",
                    "必须 plan_to_scheme 后再写/改 examples 风格脚本",
                ],
                {"base": aq_base, "moe_no_examples": aq_bonus},
                "见 22-auto-quantize.md → skills/scripts/plan_to_scheme.py",
            )
        )

    if p["moe_adapter"]:
        for mid, kind, label, hint in [
            (
                "MoE-Quant:仅路由专家_W4A8",
                "w4a8",
                "仅路由专家 W4A8",
                "cd utils/MoE-Quant && tests/run_*w4a8*.sh  # --quantize_only_experts",
            ),
            (
                "MoE-Quant:混合_W4A8",
                "w4a8",
                "混合 W4A8",
                "cd utils/MoE-Quant && quant.py --bits 4 --force_int8 ... && pack A8",
            ),
            (
                "MoE-Quant:W4A16",
                "w4",
                "W4A16",
                "cd utils/MoE-Quant && quant.py --bits 4 && pack --activation-bits 16",
            ),
        ]:
            mq_base = _F("moe_quant", "base")
            mq_reg = _F("moe_quant", "registry")
            bd = {"base": mq_base, "registry": mq_reg}
            score = mq_base + mq_reg
            pros = [
                "registry 命中 %s" % p["moe_adapter"],
                "档位术语：%s（见 13 §3.1）" % label,
            ]
            cons = [
                "两阶段 quant+pack；工作目录须在 MoE-Quant",
                "adapter validate_* 可能锁定 bits/ignore",
            ]
            if os.path.isdir(MOE_RUN_DIR):
                try:
                    if any(
                        n.startswith("run_") and "w4a8" in n and n.endswith(".sh")
                        for n in os.listdir(MOE_RUN_DIR)
                    ):
                        pros.append("存在 tests/run_*w4a8*.sh 可参考")
                        v = _F("moe_quant", "run_script")
                        bd["run_script"] = v
                        score += v
                except OSError:
                    pass
            if p["topology"] != "moe":
                v = _F("moe_quant", "non_moe_topology")
                score += v
                cons.append("拓扑非 MoE，不应走 MoE-Quant")
                bd["non_moe_topology"] = v
            tb = _target_bonus(p["target"], kind)
            bd["target"] = tb
            score += tb
            cands.append(
                _cand(mid, "MoE-Quant", _clamp(score), pros, cons, bd, hint)
            )
    else:
        cands.append(
            _cand(
                "MoE-Quant:*",
                "MoE-Quant",
                0,
                [],
                ["config 未命中 MoE-Quant registry"],
                {"hard_gate": 0},
                "需新增 adapter 并注册（13 §6）",
                excluded=True,
                exclude_reason="registry_miss",
            )
        )

    cands.append(
        _cand(
            "slimquant_ptq",
            "llmc",
            0,
            [],
            ["skill 禁止推荐/使用"],
            {"hard_gate": 0},
            "(forbidden)",
            excluded=True,
            exclude_reason="slimquant_banned",
        )
    )

    live = [c for c in cands if not c["excluded"]]
    dead = [c for c in cands if c["excluded"]]
    live.sort(key=lambda c: (-c["score"], c["method_id"]))
    return live + dead


def render_markdown(p, cands, min_score, top):
    lines = []
    lines.append("# 量化方法匹配度报告")
    lines.append("")
    lines.append("- **model**: `%s`" % p["model"])
    lines.append(
        "- **tags**: model_type=`%s` arch=`%s` topology=`%s` vl=`%s` "
        "weight=`%s` size_gb=`%s` target=`%s` history=`%s`"
        % (
            p["model_type"],
            p["architectures"],
            p["topology"],
            p["is_vl"],
            p["weight"],
            p["size_gb"],
            p["target"],
            p["has_quant_history"],
        )
    )
    if p["examples_hits"]:
        eh = ", ".join(
            ("%s[%s]" % (x[0], x[1]) if isinstance(x, (list, tuple)) else str(x))
            for x in p["examples_hits"]
        )
    else:
        eh = "∅"
    lines.append("- **examples_hits**: %s" % eh)
    lines.append("- **moe_adapter**: %s" % (p["moe_adapter"] or "∅"))
    lines.append("- **yaml_schemes**: %s" % (p["yaml_schemes"] or "∅"))
    if p["notes"]:
        lines.append("- **notes**: %s" % "; ".join(p["notes"]))
    lines.append("")
    lines.append(
        "> 匹配度 = 工程可行性分数（0–100），**不是**精度预测。"
        "交付仍须 `04` §3.2（ΔNLL < 0.1）。客户确认前不得开跑。"
    )
    lines.append("")
    lines.append(
        "| 排名 | 匹配度 | 方法 | 路径 | 优势 | 劣势 | 命令骨架 |"
    )
    lines.append("|------|--------|------|------|------|------|----------|")
    shown = [
        c for c in cands if (not c["excluded"]) and c["score"] >= min_score
    ]
    if top > 0:
        shown = shown[:top]
    for i, c in enumerate(shown, 1):
        pros = "<br>".join("• %s" % x for x in c["pros"]) or "—"
        cons = "<br>".join("• %s" % x for x in c["cons"]) or "—"
        lines.append(
            "| %s | **%s** | `%s` | %s | %s | %s | `%s` |"
            % (
                i,
                c["score"],
                c["method_id"],
                c["path"],
                pros,
                cons,
                c["command_hint"],
            )
        )
    excl = [c for c in cands if c["excluded"]]
    if excl:
        lines.append("")
        lines.append("## 已排除")
        for c in excl:
            lines.append(
                "- `%s`: %s — %s"
                % (c["method_id"], c["exclude_reason"], "; ".join(c["cons"]))
            )
    lines.append("")
    lines.append("## 分数拆解（前 5）")
    for c in shown[:5]:
        parts = ", ".join(
            "%s=%+d" % (k, v) for k, v in sorted(c["breakdown"].items())
        )
        lines.append("- `%s` (%s): %s" % (c["method_id"], c["score"], parts))
    return "\n".join(lines) + "\n"


def main(argv=None):
    global RUBRIC
    args = parse_args(argv)
    try:
        RUBRIC = load_rubric(args.rubric)
        profile = build_profile(args)
    except IOError as e:
        sys.stderr.write("error: %s\n" % e)
        return 2
    except ValueError as e:
        sys.stderr.write("error: %s\n" % e)
        return 2
    except yaml.YAMLError as e:
        sys.stderr.write("error: rubric yaml: %s\n" % e)
        return 2
    except KeyError as e:
        sys.stderr.write("error: rubric missing key %s\n" % e)
        return 2

    cands = score_candidates(profile)
    if args.format == "json":
        payload = {
            "profile": {
                "model": profile["model"],
                "model_type": profile["model_type"],
                "architectures": profile["architectures"],
                "topology": profile["topology"],
                "is_vl": profile["is_vl"],
                "weight": profile["weight"],
                "size_gb": profile["size_gb"],
                "target": profile["target"],
                "has_quant_history": profile["has_quant_history"],
                "examples_hits": profile["examples_hits"],
                "moe_adapter": profile["moe_adapter"],
                "yaml_schemes": profile["yaml_schemes"],
                "notes": profile["notes"],
            },
            "candidates": cands,
            "disclaimer": "score is engineering match 0-100, not predicted ΔNLL",
        }
        sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    else:
        sys.stdout.write(
            render_markdown(profile, cands, args.min_score, args.top)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
