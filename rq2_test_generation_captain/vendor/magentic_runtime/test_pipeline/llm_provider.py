#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv
from openai import OpenAI

import config as cfg_mod
import paths
from nlp_testcase import LLMInterface

CAPABILITY_HINTS = {
    "web_retrieval": "mention using publicly accessible online sources (e.g. \"use publicly accessible online sources\", \"locate ... online\", \"find ... on the web\")",
    "document_inspection": "mention inspecting the internal content of a document, not just its title or landing page (e.g. \"inspect the internal content of the document\")",
    "executable_computation": "mention performing an actual calculation or executable transformation (e.g. \"compute\", \"calculate\", \"derive the result\")",
    "verification": "mention confirming the result with independent evidence (e.g. \"verify\", \"cross-check\", \"confirm with an independent source\")",
    "data_extraction": "mention extracting specific information or structured data from a source (e.g. \"extract the specific information\")",
    "data_analysis": "mention analyzing/comparing/aggregating the extracted data (e.g. \"analyze\", \"compare\", \"aggregate\")",
    "code_generation": "mention producing the code needed for the task (e.g. \"produce the code\", \"write a script\")",
}

SYSTEM_PROMPT = (
    "You are an expert designer of natural-language test tasks for evaluating multi-agent systems (MAS).\n"
    "You receive an internal behavioral specification (abstract path semantics). You must convert it into a "
    "user-facing task that a real user would ask, such that solving the task NATURALLY requires the specified "
    "behaviors, WITHOUT revealing any internal implementation detail of the system.\n"
    "Hard rules:\n"
    "1. Output exactly ONE natural-language task. No numbering, no markdown, no extra commentary. A block of "
    "verbatim literal data copied under rule 8 is part of the task and is allowed.\n"
    "2. The task must require ALL hard capabilities and satisfy ALL hard constraints listed in the spec.\n"
    "3. Use ONLY the concrete facts/entities from the provided grounded seed. You may paraphrase the wording, "
    "but never invent new facts or values.\n"
    "4. Never mention internal identifiers: agent names, tool names, rule ids, states, or the testing objective. "
    "Refer to files/artifacts generically or by the provided file path only.\n"
    "5. Never reveal or hint at the expected answer (you do not have it).\n"
    "6. The task must ask for an answer of the specified type/format.\n"
    "7. Output a natural, fluent task of 20-150 words. Keep it as one paragraph, except that a verbatim block "
    "of literal data copied under rule 8 may follow the paragraph.\n"
    "8. Copy every piece of literal data in the grounded seed VERBATIM: lists/arrays, quoted strings, "
    "identifiers, file names, URLs, tables, code and numeric values. Never summarize, shorten, round, reformat "
    "or omit them - the task becomes unsolvable without them."
)


def get_api_key(api_key_env: str) -> str:
    load_dotenv(paths.ENV_PATH)
    key = os.environ.get(api_key_env, "").strip().strip("'").strip('"')
    if not key:
        raise RuntimeError(f"environment variable {api_key_env} is not set (expected from .env)")
    return key


def build_openai_client(cfg: Dict[str, Any], usage: str) -> Tuple[OpenAI, Dict[str, Any]]:
    st = cfg_mod.resolve_llm(cfg, usage)
    client = OpenAI(
        api_key=get_api_key(st["api_key_env"]),
        base_url=st["base_url"],
        max_retries=st.get("max_retries", 2),
    )
    return client, st


def chat_once(cfg: Dict[str, Any], usage: str, system: str, user: str,
              force_json: bool = False) -> str:
    client, st = build_openai_client(cfg, usage)
    kwargs: Dict[str, Any] = {}
    if force_json:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(
        model=st["model"],
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=st["temperature"],
        max_tokens=st["max_tokens"],
        **kwargs,
    )
    return (resp.choices[0].message.content or "").strip()


def build_agent_client(cfg: Dict[str, Any]):
    from autogen_core.models import ModelFamily, ModelInfo
    from autogen_ext.models.openai import OpenAIChatCompletionClient

    st = cfg_mod.resolve_llm(cfg, "mas")
    structured = bool(st.get("structured_output", False))
    client = OpenAIChatCompletionClient(
        model=st["model"],
        api_key=get_api_key(st["api_key_env"]),
        base_url=st["base_url"],
        temperature=st["temperature"],
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            structured_output=structured,
            family=ModelFamily.UNKNOWN,
        ),
    )
    from llm_usage_tracker import InstrumentedModelClient

    client = InstrumentedModelClient(client)
    return client, st


def _build_user_prompt(spec: Dict[str, Any]) -> str:
    hard_caps = list(spec.get("hard_capabilities", []))
    soft_caps = list(spec.get("soft_capabilities", []))
    seq = list(spec.get("capability_sequence", []))
    domains = list(spec.get("domain_requirements", [])) or list(spec.get("grounded_domains", []))
    slots = spec.get("grounded_slots", {}) or {}
    deps = spec.get("dependencies", []) or []
    constraints = spec.get("constraints", []) or []
    oracle_req = spec.get("oracle_requirement", {}) or {}
    ans_type = (oracle_req.get("required_format")
                or oracle_req.get("answer_type")
                or spec.get("expected_output_type")
                or "short answer")

    hard_constraint_lines = [
        f"- [{c.get('strength')}] {c.get('text')}"
        for c in constraints
        if c.get("strength") == "hard"
    ]

    lines: List[str] = ["=== GOAL ===", str(spec.get("goal"))]
    lines += ["", "=== HARD CAPABILITIES (must all be required by the task) ==="]
    lines += [f"- {c}" for c in hard_caps] or ["(none)"]
    lines += ["", "=== SOFT CAPABILITIES (reflect when possible) ==="]
    lines += [f"- {c}" for c in soft_caps] or ["(none)"]
    if seq:
        lines += ["", "=== CAPABILITY SEQUENCE (preserve order when linguistically useful) ==="]
        lines += [" -> ".join(seq)]
    if domains:
        lines += ["", "=== DOMAIN ===", ", ".join(domains)]
    if deps:
        lines += ["", "=== DEPENDENCIES (must be preserved) ==="]
        lines += [f"- {d.get('description')}" for d in deps if d.get("description")]
    lines += ["", "=== GROUNDED SEED (concrete content; use these facts only; copy literal data verbatim) ==="]
    lines.append(json.dumps(slots, ensure_ascii=False))
    lines += ["", "=== HARD CONSTRAINTS (the task MUST satisfy these) ==="]
    lines += hard_constraint_lines or ["(none)"]
    lines += ["", "=== REQUIRED ANSWER TYPE ===", str(ans_type)]
    lines += ["", "=== HOW TO REFLECT EACH CAPABILITY IN NATURAL LANGUAGE ==="]
    for c in hard_caps + [s for s in soft_caps if s not in hard_caps]:
        hint = CAPABILITY_HINTS.get(c)
        if hint:
            lines.append(f"- {c}: {hint}")
    lines += [
        "",
        "End the task with a request for the answer in the required format, e.g. "
        f"'Return the result as {ans_type}.'",
        "The request must also require the answer on its own: give only the value, with no "
        "explanation, no extra commentary and no Markdown decoration around it.",
    ]
    return "\n".join(lines)


def _clean_candidate(text: str) -> str:
    t = text.strip()
    while t.startswith("```"):
        t = t.split("\n", 1)[-1].strip()
    while t.endswith("```"):
        t = t.rsplit("\n", 1)[0].strip()
    for pfx in ("1.", "1)", "1、", "- ", "• "):
        if t.startswith(pfx):
            t = t[len(pfx):].strip()
            break
    if t.startswith('"') and t.endswith('"'):
        t = t[1:-1].strip()
    return t


class ConfiguredLLM(LLMInterface):

    def __init__(self, cfg: Dict[str, Any], usage: str = "question_generation"):
        self.cfg = cfg
        self.usage = usage
        self.client, self.settings = build_openai_client(cfg, usage)
        self.model = self.settings["model"]
        self.temperature = self.settings["temperature"]
        self.max_tokens = self.settings["max_tokens"]

    @property
    def provider(self) -> str:
        return self.settings["provider"]

    def synthesize(self, spec_for_llm: Dict[str, Any], n_candidates: int) -> List[str]:
        user = _build_user_prompt(spec_for_llm)
        out: List[str] = []
        truncated = 0
        for _ in range(max(1, int(n_candidates))):
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            choice = resp.choices[0]
            if choice.finish_reason == "length":
                truncated += 1
                continue
            text = _clean_candidate((choice.message.content or "").strip())
            if text:
                out.append(text)
        if not out:
            raise RuntimeError(
                f"{self.provider}/{self.model} returned no usable candidates"
                + (f"({truncated} candidates truncated because max_tokens was exhausted; increase max_tokens for this usage point)"
                   if truncated else "")
            )
        return out
