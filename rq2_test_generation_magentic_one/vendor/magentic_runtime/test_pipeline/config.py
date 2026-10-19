#!/usr/bin/env python3
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

import paths

DEFAULTS: Dict[str, Any] = {
    "llm": {
        "active": "deepseek",
        "providers": {
            "deepseek": {
                "base_url": "https://api.deepseek.com",
                "model": "deepseek-flash",
                "api_key_env": "DEEPSEEK_API_KEY",
                "temperature": 0.7,
                "max_tokens": 16000,
                "max_retries": 2,
            }
        },
        "usages": {
            "question_generation": {"provider": None, "temperature": 0.7, "max_tokens": 16000},
            "mas": {"provider": None, "temperature": 0.0, "max_tokens": 16000},
            "validation": {"provider": None, "temperature": 0.0, "max_tokens": 16000},
        },
    },
    "search": {
        "num_tests": 5,
        "seed": 42,
        "pop_size": 40,
        "generations": 20,
        "crossover_rate": 0.80,
        "mutation_rate": 0.35,
        "max_path_len": 30,
        "path_gen_retries": 30,
        "terminal_template_weight": 0.20,
        "k_neighbors": 5,
        "beta_alpha": 1.0,
        "beta_beta": 1.0,
    },
    "question_generation": {"n_candidates": 3, "use_llm": True, "template_fallback": True},
    "mas": {"max_turns": 20},
    "sandbox": {"mode": "docker", "image": "mas-sandbox:latest", "timeout": 180},
    "logging": {"level": "INFO"},
    "validation": {"mast": {"rules": True, "llm_judge": True}},
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    p = Path(path or paths.CONFIG_PATH)
    user_cfg: Dict[str, Any] = {}
    if p.exists():
        loaded = yaml.safe_load(p.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            user_cfg = loaded
    cfg = _deep_merge(DEFAULTS, user_cfg)
    cfg["_source"] = str(p)
    return cfg


def get(cfg: Dict[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def resolve_llm(cfg: Dict[str, Any], usage: str) -> Dict[str, Any]:
    llm_cfg = cfg.get("llm", {}) or {}
    providers = llm_cfg.get("providers", {}) or {}
    active = llm_cfg.get("active") or "deepseek"
    usage_cfg = (llm_cfg.get("usages", {}) or {}).get(usage, {}) or {}

    provider_name = usage_cfg.get("provider") or active
    if provider_name not in providers:
        raise KeyError(
            f"provider '{provider_name}' not found in config (usages.{usage} / llm.active)."
            f"registered: {sorted(providers)}"
        )
    provider = dict(providers[provider_name])

    out: Dict[str, Any] = {
        "usage": usage,
        "provider": provider_name,
        "base_url": provider.get("base_url"),
        "model": provider.get("model"),
        "api_key_env": provider.get("api_key_env"),
        "temperature": provider.get("temperature", 0.7),
        "max_tokens": provider.get("max_tokens", 16000),
        "max_retries": provider.get("max_retries", 2),
        "structured_output": bool(provider.get("structured_output", False)),
    }
    for key in ("base_url", "model", "api_key_env", "temperature", "max_tokens",
                "max_retries", "structured_output"):
        if usage_cfg.get(key) is not None:
            out[key] = usage_cfg[key]
    return out


def _pick(cli_value: Any, cfg_value: Any, default: Any) -> Any:
    if cli_value is not None:
        return cli_value
    if cfg_value is not None:
        return cfg_value
    return default


def resolve_params(cfg: Dict[str, Any], args) -> Dict[str, Any]:
    s = cfg.get("search", {}) or {}
    q = cfg.get("question_generation", {}) or {}
    sb = cfg.get("sandbox", {}) or {}
    mas = cfg.get("mas", {}) or {}

    d = DEFAULTS["search"]
    dq = DEFAULTS["question_generation"]
    dmas = DEFAULTS["mas"]

    use_llm = bool(_pick(None if args.no_llm else args.use_llm, q.get("use_llm"), dq["use_llm"]))
    if args.no_llm:
        use_llm = False

    return {
        "num_tests": _pick(args.num_tests, s.get("num_tests"), d["num_tests"]),
        "seed": _pick(args.seed, s.get("seed"), d["seed"]),
        "pop_size": _pick(args.pop_size, s.get("pop_size"), d["pop_size"]),
        "generations": _pick(args.generations, s.get("generations"), d["generations"]),
        "crossover_rate": _pick(args.crossover_rate, s.get("crossover_rate"), d["crossover_rate"]),
        "mutation_rate": _pick(args.mutation_rate, s.get("mutation_rate"), d["mutation_rate"]),
        "max_path_len": _pick(args.max_path_len, s.get("max_path_len"), d["max_path_len"]),
        "path_gen_retries": _pick(args.path_gen_retries, s.get("path_gen_retries"), d["path_gen_retries"]),
        "terminal_template_weight": _pick(
            args.terminal_template_weight, s.get("terminal_template_weight"), d["terminal_template_weight"]
        ),
        "k_neighbors": _pick(args.k_neighbors, s.get("k_neighbors"), d["k_neighbors"]),
        "beta_alpha": _pick(args.beta_alpha, s.get("beta_alpha"), d["beta_alpha"]),
        "beta_beta": _pick(args.beta_beta, s.get("beta_beta"), d["beta_beta"]),
        "n_candidates": _pick(args.n_candidates, q.get("n_candidates"), dq["n_candidates"]),
        "use_llm": use_llm,
        "template_fallback": bool(q.get("template_fallback", dq["template_fallback"])),
        "sandbox_mode": str(
            _pick(getattr(args, "sandbox_mode", None), sb.get("mode"),
                  DEFAULTS["sandbox"]["mode"])).lower(),
        "sandbox_image": _pick(None, sb.get("image"), DEFAULTS["sandbox"]["image"]),
        "sandbox_timeout": _pick(args.sandbox_timeout, sb.get("timeout"), DEFAULTS["sandbox"]["timeout"]),
        "mas_max_turns": _pick(args.mas_max_turns, mas.get("max_turns"), dmas["max_turns"]),
        "mast_rules": bool(get(cfg, "validation.mast.rules", True)),
        "mast_llm_judge": bool(get(cfg, "validation.mast.llm_judge", True)) and not args.no_mast_judge,
        "model": Path(args.model) if args.model else paths.MODEL_PATH,
    }
