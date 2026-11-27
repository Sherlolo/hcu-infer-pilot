#!/usr/bin/env python3
"""Convert ``autoquantize_plan.json`` into a role-level ``autoquantize_scheme.json``.

Role assignment uses built-in MoE fnmatch patterns (override with ``--profile``).
Per-role weight atom is the **majority** bit-width among ``module_bits``;
minority layers go under ``_exceptions``.

Atom id mapping (built-in labels)::

    weight 4 + channel  → w4_per_channel
    weight 4 + group    → w4_per_group_<N>  (N from --group-size, default 128)
    weight 8            → w8_per_channel
    act (optional)      → act_w8_per_token_dynamic when policy says A8

Requires Python >= 3.8. Example from repo root::

    python3 ./skills/scripts/plan_to_scheme.py \\
        --plan /path/to/autoquantize_plan.json \\
        --out  /path/to/autoquantize_scheme.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from fnmatch import fnmatch
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")

# Default MoE role → fnmatch (name matching only). Unmatched modules stay unmapped.
DEFAULT_ROLE_PATTERNS: Dict[str, List[str]] = {
    "routed": [
        "*mlp.experts",
        "*block_sparse_moe.experts",
        "*moe.experts",
        "*mlp.experts.*",
        "*block_sparse_moe.experts.*",
        "*moe.experts.*",
    ],
    "shared": [
        "*mlp.shared_expert.*",
        "*mlp.shared_experts.*",
    ],
    "attn": [
        "*self_attn.q_proj",
        "*self_attn.k_proj",
        "*self_attn.v_proj",
        "*self_attn.o_proj",
        "*self_attn.q_a_proj",
        "*self_attn.q_b_proj",
        "*self_attn.kv_a_proj_with_mqa",
        "*self_attn.kv_b_proj",
    ],
    "dense_mlp": [
        "*mlp.gate_proj",
        "*mlp.up_proj",
        "*mlp.down_proj",
    ],
    "router": [
        "*mlp.gate",
        "*mlp.gate.*",
        "*router*",
        "*block_sparse_moe.gate",
        "*block_sparse_moe.gate.*",
        "*moe.gate",
        "*moe.gate.*",
    ],
    "shared_gate": [
        "*mlp.shared_expert_gate",
        "*mlp.shared_expert_gate.*",
    ],
}

# Labels written into autoquantize_scheme.json (translate to backend config later).
DEFAULT_WO_ATOMS = frozenset(
    {"w8_per_channel", "w4_per_channel", "w4_per_group_128"}
)
DEFAULT_WA_ATOMS = frozenset({"act_w8_per_token_dynamic"})


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compress_ints(values: Iterable[int]) -> List[str]:
    xs = sorted({int(v) for v in values})
    if not xs:
        return []
    out: List[str] = []
    start = prev = xs[0]
    for x in xs[1:]:
        if x == prev + 1:
            prev = x
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = x
    out.append(str(start) if start == prev else f"{start}-{prev}")
    return out


def matching_roles(
    qualified_name: str,
    role_patterns: Mapping[str, Sequence[str]],
) -> List[str]:
    hits: List[str] = []
    for role, pats in role_patterns.items():
        if any(fnmatch(qualified_name, p) for p in pats):
            hits.append(role)
    return hits


def _as_int_bits(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError(f"unexpected bool bit width: {value!r}")
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError(f"bit width must be int, got {type(value).__name__}: {value!r}")


def _layer_index(name: str) -> Optional[int]:
    m = _LAYER_RE.search(name)
    return int(m.group(1)) if m else None


def _profile_patterns(profile: Mapping[str, Any]) -> Dict[str, List[str]]:
    src = profile.get("patterns") or profile.get("include") or {}
    if not isinstance(src, dict):
        raise ValueError("profile patterns must be an object")
    out: Dict[str, List[str]] = {}
    for role, pats in src.items():
        if not isinstance(pats, (list, tuple)):
            raise ValueError(f"profile patterns[{role!r}] must be a list")
        out[str(role)] = [str(p) for p in pats]
    return out


def _atom_ids_available(atoms: Mapping[str, Any]) -> Tuple[set, set]:
    wo = set((atoms.get("wo") or {}).keys())
    wa = set((atoms.get("wa") or {}).keys())
    return wo, wa


def weight_atom_for_bits(
    bits: int,
    *,
    weight_granularity: str,
    group_size: int,
    wo_ids: set,
) -> str:
    if bits == 8:
        atom = "w8_per_channel"
    elif bits == 4:
        gran = (weight_granularity or "channel").lower()
        if gran in {"group", "per_group"}:
            atom = (
                "w4_per_group_128"
                if group_size == 128
                else f"w4_per_group_{group_size}"
            )
            if atom not in wo_ids and "w4_per_group_128" in wo_ids:
                atom = "w4_per_group_128"
        else:
            atom = "w4_per_channel"
            if atom not in wo_ids and "w4_per_channel_slim" in wo_ids:
                atom = "w4_per_channel_slim"
    else:
        raise ValueError(
            f"unsupported weight bits={bits}; only 4/8 map to atoms today"
        )
    # Built-in set lists canonical ids; allow w4_per_group_<N> for non-128.
    if atom not in wo_ids and not (
        atom.startswith("w4_per_group_") and "w4_per_group_128" in wo_ids
    ):
        raise ValueError(
            f"mapped weight atom {atom!r} not in known wo ids={sorted(wo_ids)}"
        )
    return atom


def act_atom_from_policy(
    policy: Mapping[str, Any],
    *,
    enable_a8: bool,
    wa_ids: set,
) -> Optional[str]:
    if not enable_a8:
        return None
    ab = policy.get("activation_bits")
    if ab != 8:
        return None
    atom = "act_w8_per_token_dynamic"
    if atom not in wa_ids:
        raise ValueError(
            f"A8 requested but {atom!r} missing from known wa={sorted(wa_ids)}"
        )
    return atom


def _majority_bits(counter: Counter) -> int:
    if not counter:
        raise ValueError("empty bit counter")
    # Tie-break: prefer fewer bits (more aggressive) then lower numeric.
    best = sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    return int(best)


def _default_scheme_name(plan: Mapping[str, Any], *, a8: bool) -> str:
    model = plan.get("model") or {}
    model_type = str(model.get("model_type") or "model").upper().replace("-", "_")
    pol = plan.get("policy") or {}
    gran = str(pol.get("weight_granularity") or "channel").lower()
    wtag = "W4C" if gran in {"channel", "per_channel"} else "W4G"
    atag = "A8" if a8 else "A16"
    return f"MOE_AUTOQ_{model_type}_{wtag}_SA_MIX_{atag}"


def build_scheme_entry(
    plan: Mapping[str, Any],
    *,
    role_patterns: Mapping[str, Sequence[str]],
    wo_ids: set,
    wa_ids: set,
    scheme_name: str,
    enable_a8: bool,
    group_size: int,
    skip_roles: Sequence[str] = ("linear_attn", "embed", "lm_head"),
) -> Dict[str, Any]:
    if plan.get("schema") != "one-click-quant.auto_quantize":
        raise ValueError(
            f"unexpected schema {plan.get('schema')!r}; "
            "expected 'one-click-quant.auto_quantize'"
        )
    module_bits = plan.get("module_bits")
    if not isinstance(module_bits, dict) or not module_bits:
        raise ValueError("plan missing non-empty module_bits")

    policy = plan.get("policy") or {}
    weight_granularity = str(policy.get("weight_granularity") or "channel")
    act_id = act_atom_from_policy(policy, enable_a8=enable_a8, wa_ids=wa_ids)
    a8 = act_id is not None

    skip = frozenset(skip_roles)
    role_bits: Dict[str, Counter] = defaultdict(Counter)
    role_layer_bits: Dict[str, Dict[int, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    unmapped: Counter = Counter()
    ambiguous: List[Dict[str, Any]] = []

    for name, raw_bits in module_bits.items():
        bits = _as_int_bits(raw_bits)
        hits = [r for r in matching_roles(name, role_patterns) if r not in skip]
        if not hits:
            unmapped[f"{patternize(name)}@{bits}"] += 1
            continue
        if len(hits) > 1:
            ambiguous.append({"name": name, "roles": hits, "bits": bits})
            role = "routed" if "routed" in hits else hits[0]
        else:
            role = hits[0]
        role_bits[role][bits] += 1
        layer_i = _layer_index(name)
        if layer_i is not None:
            role_layer_bits[role][layer_i][bits] += 1

    if not role_bits:
        raise ValueError(
            "no module_bits matched profile patterns; check --profile vs plan names"
        )

    roles_out: Dict[str, Dict[str, Any]] = {}
    majority: Dict[str, Any] = {}
    exceptions: Dict[str, Any] = {}

    for role in sorted(role_bits):
        counter = role_bits[role]
        maj = _majority_bits(counter)
        w_atom = weight_atom_for_bits(
            maj,
            weight_granularity=weight_granularity,
            group_size=group_size,
            wo_ids=wo_ids,
        )
        role_act = None if role in ("router", "shared_gate") else act_id
        roles_out[role] = {"weight": w_atom, "act": role_act}
        majority[role] = {
            "bits": maj,
            "weight": w_atom,
            "counts": {str(b): counter[b] for b in sorted(counter)},
        }

        bad_layers: Dict[int, Dict[str, int]] = {}
        for layer_i, lc in role_layer_bits[role].items():
            layer_maj = _majority_bits(lc)
            if layer_maj != maj:
                bad_layers[layer_i] = {str(b): lc[b] for b in sorted(lc)}
        if bad_layers:
            exceptions[role] = {
                "role_majority_bits": maj,
                "layers_not_majority": compress_ints(bad_layers.keys()),
                "per_layer_bit_counts": {
                    str(k): bad_layers[k] for k in sorted(bad_layers)
                },
            }

    excluded = plan.get("excluded_modules") or {}
    excluded_by_pattern: Counter = Counter()
    for name in excluded:
        excluded_by_pattern[patternize(str(name))] += 1

    model = plan.get("model") or {}
    entry: Dict[str, Any] = {
        "family": "moe",
        "a8": a8,
        "allowed_suffixes": [],
        "global_weight_enabled": False,
        "roles": roles_out,
        "_from_autoquantize": {
            "plan_schema": plan.get("schema"),
            "method": plan.get("method"),
            "modelopt_version": plan.get("modelopt_version"),
            "model_id": model.get("model_id"),
            "model_type": model.get("model_type"),
            "effective_bits_target": plan.get("effective_bits_target"),
            "actual_effective_bits": plan.get("actual_effective_bits"),
            "policy": policy,
            "note": (
                "roles use majority bit-width over module_bits; "
                "per-layer mixes cannot be expressed under role_specs — "
                "see _exceptions before executing quantization."
            ),
        },
        "_majority": majority,
        "_exceptions": exceptions,
        "_unmapped_module_bits": {
            "by_pattern_bits": dict(sorted(unmapped.items())),
            "n_ambiguous": len(ambiguous),
            "ambiguous_sample": ambiguous[:10],
        },
        "_excluded_modules": {
            "n": len(excluded),
            "by_pattern": dict(sorted(excluded_by_pattern.items())),
        },
    }
    return {scheme_name: entry}


def patternize(name: str) -> str:
    name = re.sub(r"\.layers\.\d+\.", ".layers.*.", name)
    name = re.sub(r"\.experts\.\d+\.", ".experts.*.", name)
    return name


def scheme_style_document(
    fragment: Mapping[str, Any],
    *,
    strip_private: bool,
) -> Dict[str, Any]:
    """Wrap like scheme.json; optionally drop ``_`` diagnostic keys inside entry."""
    doc: Dict[str, Any] = {
        "_comment": (
            "Generated from autoquantize_plan.json by plan_to_scheme.py. "
            "Role weights = majority vote over module_bits; see each entry's "
            "_exceptions for layers that differ. Review before quantization."
        ),
    }
    for name, entry in fragment.items():
        if not strip_private:
            doc[name] = entry
            continue
        clean = {
            k: v
            for k, v in entry.items()
            if not str(k).startswith("_")
        }
        doc[name] = clean
    return doc


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan",
        type=Path,
        required=True,
        help="autoquantize_plan.json from utils.auto_quantize",
    )
    parser.add_argument(
        "--profile",
        type=Path,
        default=None,
        help="optional JSON with patterns{role:[fnmatch,...]}; default=built-in MoE",
    )
    parser.add_argument(
        "--atoms",
        type=Path,
        default=None,
        help="optional JSON with wo/wa keys; default=built-in atom id set",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="output autoquantize_scheme.json path",
    )
    parser.add_argument(
        "--scheme-name",
        default=None,
        help="scheme key; default derived from model_type + policy",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=128,
        help="used only when policy weight_granularity is group",
    )
    parser.add_argument(
        "--a8",
        dest="a8",
        action="store_true",
        default=None,
        help="force include act_w8_per_token_dynamic on non-gate roles",
    )
    parser.add_argument(
        "--no-a8",
        dest="a8",
        action="store_false",
        help="force weight-only (act=null) even if plan policy has activation_bits",
    )
    parser.add_argument(
        "--strip-private",
        action="store_true",
        help="omit _from_autoquantize / _exceptions / … (pure scheme shape)",
    )
    parser.add_argument(
        "--merge-into",
        type=Path,
        default=None,
        help="optional existing scheme.json to merge the new key into (writes --out)",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    plan = _load_json(args.plan)
    if args.profile is not None:
        role_patterns = _profile_patterns(_load_json(args.profile))
    else:
        role_patterns = {k: list(v) for k, v in DEFAULT_ROLE_PATTERNS.items()}

    if args.atoms is not None:
        wo_ids, wa_ids = _atom_ids_available(_load_json(args.atoms))
    else:
        wo_ids, wa_ids = set(DEFAULT_WO_ATOMS), set(DEFAULT_WA_ATOMS)

    policy = plan.get("policy") or {}
    if args.a8 is None:
        enable_a8 = int(policy.get("activation_bits") or 0) == 8
    else:
        enable_a8 = bool(args.a8)

    scheme_name = args.scheme_name or _default_scheme_name(plan, a8=enable_a8)
    fragment = build_scheme_entry(
        plan,
        role_patterns=role_patterns,
        wo_ids=wo_ids,
        wa_ids=wa_ids,
        scheme_name=scheme_name,
        enable_a8=enable_a8,
        group_size=int(args.group_size),
    )

    if args.merge_into is not None:
        base_doc = _load_json(args.merge_into)
        if not isinstance(base_doc, dict):
            raise ValueError("--merge-into must be a JSON object")
        if scheme_name in base_doc and not str(scheme_name).startswith("_"):
            raise ValueError(
                f"refusing to overwrite existing scheme key {scheme_name!r} "
                f"in {args.merge_into}; pick --scheme-name"
            )
        out_doc = dict(base_doc)
        entry = fragment[scheme_name]
        if args.strip_private:
            entry = {k: v for k, v in entry.items() if not str(k).startswith("_")}
        out_doc[scheme_name] = entry
    else:
        out_doc = scheme_style_document(fragment, strip_private=args.strip_private)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out_doc, indent=2, ensure_ascii=False) + "\n")

    entry = fragment[scheme_name]
    print(f"scheme: {scheme_name}")
    print(f"family=moe a8={entry['a8']}")
    for role, spec in entry["roles"].items():
        maj = entry["_majority"][role]
        print(
            f"  role {role}: weight={spec['weight']} act={spec['act']} "
            f"(majority_bits={maj['bits']}, counts={maj['counts']})"
        )
    if entry.get("_exceptions"):
        print("exceptions (layers ≠ role majority):")
        for role, ex in entry["_exceptions"].items():
            print(
                f"  {role}: layers {', '.join(ex['layers_not_majority'])} "
                f"(role_majority_bits={ex['role_majority_bits']})"
            )
    print(f"[wrote] {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
