#!/usr/bin/env python3
from __future__ import annotations

import abc
import copy
import json
import random
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from fragment_matcher_v0_4 import (
    FragmentMatcher,
    GenerationFailure as _FragmentFailure,
    assemble_slot_rules,
)



class GenerationFailure(RuntimeError):

    def __init__(self, reason: str, details: Optional[Dict[str, Any]] = None):
        super().__init__(reason)
        self.reason = reason
        self.details = details or {}


@dataclass
class NormalizedGene:
    rule_id: str
    event: str
    params: Dict[str, str] = field(default_factory=dict)


@dataclass
class OperationAnchor:
    operation_id: str
    operation_type: str
    raw_events: List[str]
    path_indices: List[int]
    source_entities: List[str] = field(default_factory=list)
    target_entities: List[str] = field(default_factory=list)
    pre_state: Optional[Dict[str, Any]] = None
    post_state: Optional[Dict[str, Any]] = None


@dataclass
class ContextAnchor:
    path_index: int
    event: str
    semantic_role: str
    pre_state: Optional[Dict[str, Any]] = None
    post_state: Optional[Dict[str, Any]] = None


@dataclass
class BehaviorProjection:
    operation_signature: List[str]
    operations: List[OperationAnchor]
    ordering: List[Tuple[str, str]]
    control_context: List[ContextAnchor]
    signature_policy: str
    state_replay_available: bool
    full_path: List[Dict[str, Any]]


@dataclass
class CandidateValidation:
    query: str
    accepted: bool
    score: float
    checks: Dict[str, Any]


@dataclass
class GeneratedUserTask:
    query: str
    target_path: List[Dict[str, Any]]
    behavior_projection: Dict[str, Any]
    matched_fragment: Dict[str, Any]
    grounded_instance: Dict[str, Any]
    brs: Dict[str, Any]
    candidate_validations: List[Dict[str, Any]]
    oracle: Dict[str, Any] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)



LEGACY_OP_CANON = {"web_retrieval": "web_interaction"}


def canonical_op(name: str) -> str:
    return LEGACY_OP_CANON.get(str(name), str(name))


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def normalize_path(path: Sequence[Any]) -> List[NormalizedGene]:
    out: List[NormalizedGene] = []
    for g in path:
        if isinstance(g, dict):
            params = g.get("params", {}) or {}
            if isinstance(params, list):
                params = dict(params)
            out.append(
                NormalizedGene(
                    str(g.get("rule_id", "")),
                    str(g.get("event", "")),
                    {str(k): str(v) for k, v in dict(params).items()},
                )
            )
        else:
            params = getattr(g, "params", {}) or {}
            if isinstance(params, tuple):
                params = dict(params)
            out.append(
                NormalizedGene(
                    str(getattr(g, "rule_id", "")),
                    str(getattr(g, "event", "")),
                    {str(k): str(v) for k, v in dict(params).items()},
                )
            )
    return out


def path_to_dicts(path: Sequence[Any]) -> List[Dict[str, Any]]:
    return [
        {"rule_id": g.rule_id, "event": g.event, "params": dict(g.params)}
        for g in normalize_path(path)
    ]


def event_prefix(event: str) -> str:
    return event.split("(", 1)[0].strip()


def event_arguments(event: str) -> List[str]:
    m = re.search(r"\((.*)\)", event)
    if not m:
        return []
    return [x.strip() for x in m.group(1).split(",") if x.strip()]


def _event_pattern_regex(pattern: str) -> re.Pattern:
    escaped = re.escape(pattern).replace(r"\*", r"[^,()]+")
    return re.compile(r"^" + escaped + r"$")


def _matches_event(pattern: str, event: str) -> bool:
    return bool(_event_pattern_regex(pattern).match(event))


def _source_target(event: str, params: Mapping[str, str]) -> Tuple[Optional[str], Optional[str]]:
    args = event_arguments(event)
    prefix = event_prefix(event)
    if prefix in {"DELEGATE", "SELECT_INITIAL", "SELECT_NEXT"}:
        return None, params.get("agent") or (args[0] if args else None)
    if prefix == "REPORT":
        return params.get("agent") or (args[0] if args else None), None
    if prefix == "TOOL_CALL":
        if len(args) >= 2:
            return args[0], args[1]
        return params.get("agent"), params.get("tool")
    if prefix == "TOOL_RETURN":
        if len(args) >= 2:
            return args[0], args[1]
        return params.get("tool"), params.get("agent")
    return None, None



def _try_import_runtime():
    try:
        from adaptive_search import Gene, ModelRuntime
        return Gene, ModelRuntime
    except Exception:
        pass

    here = Path(__file__).resolve()
    candidates = [
        here.parent / "core",
        here.parent.parent / "core",
        Path.cwd() / "core",
    ]
    for c in candidates:
        if c.exists() and str(c) not in sys.path:
            sys.path.insert(0, str(c))
        try:
            from adaptive_search import Gene, ModelRuntime
            return Gene, ModelRuntime
        except Exception:
            continue
    return None, None


def replay_states(path: Sequence[Any], model: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    Gene, ModelRuntime = _try_import_runtime()
    if Gene is None or ModelRuntime is None:
        return None
    try:
        runtime = ModelRuntime(model)
        genes = []
        for g in normalize_path(path):
            genes.append(Gene(g.rule_id, g.event, tuple(sorted(g.params.items()))))
        ok, _, states = runtime.simulate(genes)
        return states if ok and len(states) == len(genes) + 1 else None
    except Exception:
        return None



class TransitionSemanticMap:
    def __init__(self, path: Path):
        self.data = load_json(path)

    @staticmethod
    def family_key(model: Dict[str, Any]) -> str:
        family = str(model.get("family", ""))
        mid = str(model.get("model_id", ""))
        if family.startswith("CaptainAgent") or "CaptainAgent" in mid:
            return "CaptainAgent"
        if "Magentic-One" in family or "MagenticOne" in family or "Magentic" in mid:
            return "Magentic-One"
        raise GenerationFailure(
            "unsupported_framework_semantic_map",
            {"family": family, "model_id": mid},
        )

    def classify(self, event: str, model: Dict[str, Any]) -> Dict[str, Any]:
        key = self.family_key(model)
        spec = self.data[key]
        for pattern, operation in spec.get("operational", {}).items():
            if _matches_event(pattern, event):
                return {
                    "semantic_role": "operational",
                    "operation": canonical_op(operation),
                    "pattern": pattern,
                }
        for role in ("contextual", "administrative"):
            for pattern in spec.get(role, []):
                if _matches_event(pattern, event):
                    return {"semantic_role": role, "pattern": pattern}
        return {"semantic_role": "unsupported", "pattern": None}


class BehaviorInterpreter:

    STATE_KEYS = (
        "activeEntity",
        "executionStatus",
        "budget",
        "phase",
        "requestSatisfied",
        "progressStatus",
        "loopStatus",
        "stallCount",
    )

    def __init__(self, semantic_map: TransitionSemanticMap, signature_policy: str = "exact"):
        if signature_policy not in {"exact", "collapse_adjacent"}:
            raise ValueError("signature_policy must be exact or collapse_adjacent")
        self.semantic_map = semantic_map
        self.signature_policy = signature_policy

    @classmethod
    def _project_state(cls, s: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if s is None:
            return None
        return {k: s[k] for k in cls.STATE_KEYS if k in s}

    def interpret(self, path: Sequence[Any], model: Dict[str, Any]) -> BehaviorProjection:
        genes = normalize_path(path)
        states = replay_states(path, model)
        ops: List[OperationAnchor] = []
        context: List[ContextAnchor] = []

        for i, g in enumerate(genes):
            sem = self.semantic_map.classify(g.event, model)
            pre = self._project_state(states[i] if states else None)
            post = self._project_state(states[i + 1] if states else None)

            if sem["semantic_role"] == "operational":
                op_type = str(sem["operation"])
                source, target = _source_target(g.event, g.params)
                if (
                    self.signature_policy == "collapse_adjacent"
                    and ops
                    and ops[-1].operation_type == op_type
                ):
                    ops[-1].raw_events.append(g.event)
                    ops[-1].path_indices.append(i)
                    if source:
                        ops[-1].source_entities.append(source)
                    if target:
                        ops[-1].target_entities.append(target)
                    ops[-1].post_state = post
                else:
                    oid = f"o{len(ops) + 1}"
                    ops.append(
                        OperationAnchor(
                            operation_id=oid,
                            operation_type=op_type,
                            raw_events=[g.event],
                            path_indices=[i],
                            source_entities=[source] if source else [],
                            target_entities=[target] if target else [],
                            pre_state=pre,
                            post_state=post,
                        )
                    )
            else:
                context.append(
                    ContextAnchor(
                        path_index=i,
                        event=g.event,
                        semantic_role=sem["semantic_role"],
                        pre_state=pre,
                        post_state=post,
                    )
                )

        signature = [x.operation_type for x in ops]
        ordering = [
            (ops[i].operation_id, ops[i + 1].operation_id)
            for i in range(len(ops) - 1)
        ]
        if not signature:
            raise GenerationFailure(
                "no_task_level_operation",
                {"events": [g.event for g in genes]},
            )

        return BehaviorProjection(
            operation_signature=signature,
            operations=ops,
            ordering=ordering,
            control_context=context,
            signature_policy=self.signature_policy,
            state_replay_available=states is not None,
            full_path=path_to_dicts(path),
        )



class GroundedInstanceRepository:

    def __init__(self, instance_file: Path, data_type_file: Optional[Path] = None):
        self.instance_doc = load_json(instance_file)
        self.instances: List[Dict[str, Any]] = list(
            self.instance_doc.get("instances", [])
        )
        self.data_types = load_json(data_type_file) if data_type_file else {}
        self.excluded_instance_ids: set[str] = set()
        for inst in self.instances:
            raw = inst.get("raw_operation_sequence") or inst.get(
                "operation_signature"
            ) or []
            inst["_canonical_raw_types"] = sorted(
                {canonical_op(x) for x in raw}
            )

    def exclude_instances(self, ids: Iterable[str]) -> None:
        self.excluded_instance_ids |= {str(x) for x in ids if x}

    def eligible_instances(self) -> List[Dict[str, Any]]:
        return [
            copy.deepcopy(x)
            for x in self.instances
            if str(x.get("instance_id")) not in self.excluded_instance_ids
        ]

    @staticmethod
    def instance_covers(inst: Dict[str, Any], required_types: set[str]) -> bool:
        return required_types.issubset(set(inst.get("_canonical_raw_types", [])))

    def type_is_compatible(self, actual: str, required: str) -> bool:
        if actual == required:
            return True
        seen = set()
        cur = actual
        while cur and cur not in seen:
            seen.add(cur)
            parent = (self.data_types.get(cur) or {}).get("parent")
            if parent == required:
                return True
            cur = parent
        return False


def _bindings_to_map(raw: Any) -> Dict[str, Dict[str, Any]]:
    if isinstance(raw, dict):
        out = {}
        for slot, value in raw.items():
            if isinstance(value, dict):
                item = copy.deepcopy(value)
                item.setdefault("slot", slot)
            else:
                item = {"slot": slot, "value": value}
            out[str(slot)] = item
        return out
    out = {}
    for item in raw or []:
        if not isinstance(item, dict) or not item.get("slot"):
            continue
        out[str(item["slot"])] = copy.deepcopy(item)
    return out



class FragmentBRSBuilder:

    @staticmethod
    def _slot_types(element_meta: Dict[str, Any]) -> Dict[str, str]:
        return {
            str(r["slot"]): str(r.get("type", ""))
            for r in element_meta.get("data_roles", [])
            if r.get("slot")
        }

    def external_inputs(
        self, match: Dict[str, Any], assembly: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        consumer_slots = {str(u["right"]) for u in assembly.get("slot_unions", [])}
        external: List[Dict[str, Any]] = []
        documents: List[Dict[str, Any]] = []
        for item in match["operation_elements"]:
            pos = item["position"]
            meta = item["metadata"]
            op = meta["operation"]
            slot_types = self._slot_types(meta)
            for slot in op.get("inputs", []):
                alpha = f"{slot}@{pos}"
                if alpha in consumer_slots:
                    continue
                rec = {
                    "position": pos,
                    "operation": op["type"],
                    "slot": alpha,
                    "type": slot_types.get(slot, ""),
                }
                external.append(rec)
                if rec["type"] == "Document":
                    documents.append(rec)
        return external, documents

    @staticmethod
    def _grounding(instance: Dict[str, Any], document_artifacts: List[Dict[str, Any]]):
        bindings = _bindings_to_map(instance.get("bindings"))
        source_task = str(instance.get("source_task", "")).strip()
        if bindings:
            return {
                "mode": "explicit_slot_bindings",
                "bindings": bindings,
                "source": instance.get("source"),
                "source_id": instance.get("source_id"),
                "document_artifacts": document_artifacts,
            }
        if not source_task:
            raise GenerationFailure(
                "no_usable_grounding",
                {"instance_id": instance.get("instance_id"),
                 "reason": "instance has neither bindings nor source_task"},
            )
        return {
            "mode": "sealed_source_task",
            "source_task": source_task,
            "source": instance.get("source"),
            "source_id": instance.get("source_id"),
            "document_artifacts": document_artifacts,
        }

    def build(
        self,
        projection: BehaviorProjection,
        match: Dict[str, Any],
        assembly: Dict[str, Any],
        instance: Dict[str, Any],
        attachment_info: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        external, doc_slots = self.external_inputs(match, assembly)
        doc_artifacts: List[Dict[str, Any]] = []
        for rec in doc_slots:
            if attachment_info is None:
                raise GenerationFailure(
                    "missing_document_artifact",
                    {
                        "instance_id": instance.get("instance_id"),
                        "source_id": instance.get("source_id"),
                        "slot": rec["slot"],
                        "reason": "GAIA attachment provisioning unavailable",
                    },
                )
            doc_artifacts.append(
                {
                    "slot": rec["slot"],
                    "position": rec["position"],
                    "file_name": attachment_info.get("file_name"),
                    "sandbox_path": attachment_info.get("sandbox_path"),
                    "split": attachment_info.get("split", ""),
                }
            )
        grounding = self._grounding(instance, doc_artifacts)
        return {
            "goal_frame": copy.deepcopy(instance.get("goal_frame", {})),
            "operations": copy.deepcopy(assembly.get("operations", [])),
            "slot_unions": copy.deepcopy(assembly.get("slot_unions", [])),
            "external_inputs": external,
            "grounding": grounding,
            "control_context": [
                {
                    "event": c.event,
                    "semantic_role": c.semantic_role,
                    "pre_state": c.pre_state,
                    "post_state": c.post_state,
                }
                for c in projection.control_context
            ],
            "behavior_signature": list(projection.operation_signature),
        }



class Realizer(abc.ABC):
    @abc.abstractmethod
    def realize(
        self,
        brs: Dict[str, Any],
        operation_vocab: Dict[str, Any],
        n_candidates: int,
    ) -> List[str]:
        raise NotImplementedError


class SourceTaskRealizer(Realizer):

    def realize(self, brs, operation_vocab, n_candidates):
        task = str((brs.get("grounding") or {}).get("source_task", "")).strip()
        if not task:
            raise GenerationFailure("source_task_fallback_unavailable")
        return [task]


class LLMRealizer(Realizer):

    def __init__(self, chat: Any, temperature: float = 0.2, max_tokens: int = 16000):
        self.chat = chat
        self.temperature = temperature
        self.max_tokens = max_tokens

    @staticmethod
    def _safe_brs(brs: Dict[str, Any], operation_vocab: Dict[str, Any]) -> Dict[str, Any]:
        operations = []
        for op in brs.get("operations", []):
            op_type = op.get("type")
            operations.append(
                {
                    "position": op.get("position"),
                    "type": op_type,
                    "description": (operation_vocab.get(op_type) or {}).get(
                        "description", ""
                    ),
                    "inputs": op.get("inputs", []),
                    "outputs": op.get("outputs", []),
                }
            )
        grounding = copy.deepcopy(brs.get("grounding", {}))
        grounding.pop("reference_answer", None)
        return {
            "goal_frame": copy.deepcopy(brs.get("goal_frame", {})),
            "operations": operations,
            "slot_unions": copy.deepcopy(brs.get("slot_unions", [])),
            "grounding": grounding,
        }

    def realize(self, brs, operation_vocab, n_candidates):
        spec = self._safe_brs(brs, operation_vocab)
        system = (
            "You are a semantic-preserving surface realizer for black-box user "
            "tasks. The task semantics are fixed by a structured Behavioral "
            "Requirement Specification (BRS) assembled from reusable behavior "
            "fragments. Do NOT plan a new task and do NOT invent new goals, "
            "entities, dates, numbers, files, data dependencies, operations, or "
            "ordering. The listed operations are the target behavior: the "
            "request must be naturally solvable by performing exactly those "
            "operation types in that order, but you may not add operations. "
            "Preserve every grounded value and every relation-backed shared "
            "object. If grounding.mode is sealed_source_task, preserve every "
            "substantive fact, constraint, entity, date/number, and requested "
            "output of source_task; only surface wording may change. Refer to "
            "any grounded attachment by its file name if one is listed; its in-"
            "sandbox location is provided to the user separately and must not "
            "be guessed. Do not mention agents, tools, rule IDs, state "
            "machines, fragments, target paths, coverage, testing, or any "
            "internal implementation. Return JSON: "
            "{\"candidates\":[\"...\", ...]}."
        )
        payload = {"n_candidates": int(n_candidates), "brs": spec}
        raw = self.chat.complete(
            [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format_json=True,
        )
        parsed = self._parse_candidates(raw)
        if parsed:
            return parsed[:n_candidates]
        return []

    @staticmethod
    def _parse_candidates(raw: str) -> List[str]:
        if not raw or not str(raw).strip():
            return []
        text = str(raw).strip()
        if text.startswith("```"):
            text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
            text = re.sub(r"\n?```$", "", text).strip()
        candidates = [text]
        m = re.search(r"\{[\s\S]*\}", text)
        if m:
            candidates.append(m.group(0))
        for c in candidates:
            try:
                data = json.loads(c)
            except Exception:
                continue
            if isinstance(data, dict):
                xs = data.get("candidates")
                if isinstance(xs, list):
                    xs = [str(x).strip() for x in xs if str(x).strip()]
                    if xs:
                        return xs
        return []



def _tokens(text: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", text.casefold())


def _significant_literals(text: str) -> List[str]:
    out = re.findall(
        r"https?://\S+|10\.\d{4,9}/\S+|\b(?:19|20)\d{2}\b|\b\d{2,}\b", text
    )
    out += [x[0] or x[1] for x in re.findall(r'"([^"]{3,})"|“([^”]{3,})”', text)]
    return [x.strip(".,;:!?()[]{}") for x in out if x]


class LightweightValidator:
    def __init__(self, operation_vocab: Dict[str, Any]):
        self.operation_vocab = operation_vocab

    @staticmethod
    def _internal_terms(model: Dict[str, Any]) -> List[str]:
        structure = model.get("structure", {}) or {}
        terms = list(structure.get("agents", [])) + list(structure.get("tools", []))
        terms += [
            "target path",
            "test model",
            "transition rule",
            "coverage objective",
            "pareto front",
            "activeentity",
            "executionstatus",
        ]
        return [str(x) for x in terms if str(x).strip()]

    def _operation_positions(self, query: str, signature: Sequence[str]):
        q = query.casefold()
        positions: List[Optional[int]] = []
        evidence: List[Optional[str]] = []
        for op in signature:
            item = self.operation_vocab.get(op, {}) or {}
            aliases = [op.replace("_", " ")] + list(item.get("aliases", []))
            hits = []
            for alias in aliases:
                pos = q.find(str(alias).casefold())
                if pos >= 0:
                    hits.append((pos, alias))
            if hits:
                pos, alias = min(hits)
                positions.append(pos)
                evidence.append(alias)
            else:
                positions.append(None)
                evidence.append(None)
        evaluable = all(x is not None for x in positions)
        ordered = None
        if evaluable:
            ordered = all(
                positions[i] <= positions[i + 1] for i in range(len(positions) - 1)
            )
        return {
            "evaluable": evaluable,
            "ordered": ordered,
            "positions": positions,
            "lexical_evidence": evidence,
        }

    @staticmethod
    def _binding_values(brs: Dict[str, Any]) -> List[str]:
        grounding = brs.get("grounding", {})
        if grounding.get("mode") != "explicit_slot_bindings":
            return []
        vals = []
        for item in (grounding.get("bindings") or {}).values():
            value = item.get("surface") or item.get("value")
            if value is not None and len(str(value).strip()) >= 2:
                vals.append(str(value).strip())
        return vals

    def validate(
        self,
        query: str,
        brs: Dict[str, Any],
        model: Dict[str, Any],
        source_instance: Dict[str, Any],
    ) -> CandidateValidation:
        checks: Dict[str, Any] = {}
        qlow = query.casefold()

        leakage = [t for t in self._internal_terms(model) if t.casefold() in qlow]
        checks["internal_leakage"] = {"pass": not leakage, "terms": leakage}

        checks["unbound_placeholder"] = {
            "pass": re.search(r"\?[A-Za-z_][A-Za-z0-9_]*", query) is None
        }

        missing_bindings = [
            v for v in self._binding_values(brs) if v.casefold() not in qlow
        ]
        checks["binding_preservation"] = {
            "pass": not missing_bindings,
            "missing": missing_bindings,
            "applicable": bool(self._binding_values(brs)),
        }

        source_task = str((brs.get("grounding") or {}).get("source_task", ""))
        if source_task:
            src = set(_tokens(source_task))
            dst = set(_tokens(query))
            overlap = len(src & dst) / max(1, len(src))
            required_literals = _significant_literals(source_task)
            missing_literals = [
                x for x in required_literals if x.casefold() not in qlow
            ]
            checks["bootstrap_semantic_preservation"] = {
                "pass": (not missing_literals) and overlap >= 0.18,
                "token_recall": overlap,
                "missing_significant_literals": missing_literals,
            }
        else:
            checks["bootstrap_semantic_preservation"] = {
                "pass": True,
                "applicable": False,
            }

        op_order = self._operation_positions(query, brs.get("behavior_signature", []))
        checks["operation_lexical_order"] = {
            "pass": op_order["ordered"] is not False,
            **op_order,
            "hard_when_evaluable": True,
        }

        hard_names = [
            "internal_leakage",
            "unbound_placeholder",
            "binding_preservation",
            "bootstrap_semantic_preservation",
            "operation_lexical_order",
        ]
        accepted = all(bool(checks[k]["pass"]) for k in hard_names)
        score = (
            sum(1.0 if checks[k]["pass"] else 0.0 for k in hard_names) / len(hard_names)
        )
        return CandidateValidation(query=query, accepted=accepted, score=score, checks=checks)



AttachmentMap = Dict[str, Dict[str, Any]]


class BRSConcretizer:
    def __init__(
        self,
        library_root: Path,
        *,
        chat: Any = None,
        temperature: float = 0.2,
        max_tokens: int = 16000,
        n_surface_candidates: int = 3,
        grounding_mode: str = "bootstrap",
        signature_policy: str = "exact",
        allow_source_task_fallback: bool = True,
        matcher_seed: int = 7,
        attachment_map: Optional[AttachmentMap] = None,
    ):
        self.library_root = Path(library_root)
        fragment_file = (
            self.library_root / "task_schema_fragment_library.v0_4.json"
        )
        instance_file = self.library_root / "grounded_instance_library.bootstrap.json"
        semantic_file = self.library_root / "transition_semantic_map.json"
        op_vocab = self.library_root / "operation_vocabulary.json"
        data_types = self.library_root / "data_type_vocabulary.json"
        intent_vocab = self.library_root / "intent_vocabulary.json"

        self.matcher = FragmentMatcher(fragment_file, seed=matcher_seed)
        self.repo = GroundedInstanceRepository(
            instance_file, data_types if data_types.exists() else None
        )
        self.semantic_map = TransitionSemanticMap(semantic_file)
        self.operation_vocab = load_json(op_vocab) if op_vocab.exists() else {}
        self.intent_vocab = load_json(intent_vocab) if intent_vocab.exists() else {}
        self.signature_policy = signature_policy
        self.interpreter = BehaviorInterpreter(self.semantic_map, signature_policy)
        self.builder = FragmentBRSBuilder()
        self.validator = LightweightValidator(self.operation_vocab)
        self.n_surface_candidates = int(n_surface_candidates)
        self.grounding_mode = grounding_mode
        self.allow_source_task_fallback = allow_source_task_fallback
        self.attachment_map: AttachmentMap = dict(attachment_map or {})
        self.realizer: Realizer = (
            LLMRealizer(chat, temperature=temperature, max_tokens=max_tokens)
            if chat is not None
            else SourceTaskRealizer()
        )


    def set_attachment_map(self, attachment_map: Optional[AttachmentMap]) -> None:
        self.attachment_map = dict(attachment_map or {})


    def exclude_instance_ids(self, ids: Iterable[str]) -> None:
        self.repo.exclude_instances(ids)

    def exclude_current_model_source(self, model: Dict[str, Any]) -> List[str]:
        src = model.get("source", {}) or {}
        qid = str(src.get("question_id", "")).strip()
        dataset_file = str(src.get("dataset_file", "")).strip()
        excluded = []
        for x in self.repo.instances:
            iid = str(x.get("instance_id", ""))
            if qid and str(x.get("source_id", "")).strip() == qid:
                excluded.append(iid)
            elif dataset_file and dataset_file in str(x.get("source_file", "")):
                excluded.append(iid)
        self.exclude_instance_ids(excluded)
        return sorted(set(excluded))


    def _fragment_match(self, signature: Sequence[str]) -> Dict[str, Any]:
        try:
            match = self.matcher.choose(list(signature))
        except _FragmentFailure as e:
            raise GenerationFailure(
                "schema_gap",
                {"operation_signature": list(signature), "detail": str(e)},
            )
        match["assembly"] = assemble_slot_rules(match)
        return match

    @staticmethod
    def _match_summary(match: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "target_operations": list(match["target_operations"]),
            "relation_score": match["relation_score"],
            "relations": copy.deepcopy(match["relations"]),
            "selection_reason": match["selection_reason"],
            "operation_element_ids": [
                {"position": x["position"], "fragment_id": x["fragment_id"]}
                for x in match["operation_elements"]
            ],
        }

    def _attachment_for(self, instance: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        sid = str(instance.get("source_id") or "").strip()
        return self.attachment_map.get(sid) if sid else None

    def _usable_instances(
        self,
        projection: BehaviorProjection,
        match: Dict[str, Any],
    ) -> Tuple[List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]], List[Dict[str, Any]]]:
        required = set(projection.operation_signature)
        _, doc_slots = self.builder.external_inputs(match, match["assembly"])
        needs_document = bool(doc_slots)
        usable: List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]] = []
        rejection_log: List[Dict[str, Any]] = []
        for inst in self.repo.eligible_instances():
            if not self.repo.instance_covers(inst, required):
                rejection_log.append(
                    {
                        "instance_id": inst.get("instance_id"),
                        "reason": "element_coverage_gap",
                        "required": sorted(required),
                        "observed": inst.get("_canonical_raw_types"),
                    }
                )
                continue
            attachment_info = self._attachment_for(inst) if needs_document else None
            if needs_document and attachment_info is None:
                rejection_log.append(
                    {
                        "instance_id": inst.get("instance_id"),
                        "source_id": inst.get("source_id"),
                        "reason": "missing_document_artifact",
                        "slots": [r["slot"] for r in doc_slots],
                    }
                )
                continue
            try:
                brs = self.builder.build(
                    projection, match, match["assembly"], inst, attachment_info
                )
            except GenerationFailure as e:
                rejection_log.append(
                    {
                        "instance_id": inst.get("instance_id"),
                        "reason": e.reason,
                        "details": e.details,
                    }
                )
                continue
            usable.append((inst, attachment_info or {}, brs))
        return usable, rejection_log


    def can_concretize(
        self,
        path: Sequence[Any],
        model: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            projection = self.interpreter.interpret(path, model)
        except GenerationFailure as e:
            return {"ok": False, "reason": e.reason, "details": e.details}

        try:
            match = self._fragment_match(projection.operation_signature)
        except GenerationFailure as e:
            return {
                "ok": False,
                "reason": e.reason,
                "operation_signature": projection.operation_signature,
                "details": e.details,
            }

        usable, rejections = self._usable_instances(projection, match)
        _, doc_slots = self.builder.external_inputs(match, match["assembly"])
        return {
            "ok": bool(usable),
            "reason": None if usable else "no_usable_grounding",
            "operation_signature": projection.operation_signature,
            "fragment_match": self._match_summary(match),
            "external_document_slots": [r["slot"] for r in doc_slots],
            "usable_grounding_count": len(usable),
            "rejected_groundings": rejections[:50],
            "rejected_grounding_count": len(rejections),
        }

    def generate(
        self,
        path: Sequence[Any],
        model: Dict[str, Any],
        rng: random.Random,
    ) -> GeneratedUserTask:
        projection = self.interpreter.interpret(path, model)
        match = self._fragment_match(projection.operation_signature)
        usable_pairs, rejection_log = self._usable_instances(projection, match)
        if not usable_pairs:
            raise GenerationFailure(
                "no_usable_grounding",
                {
                    "operation_signature": projection.operation_signature,
                    "fragment_match": self._match_summary(match),
                    "rejections": rejection_log,
                },
            )

        grouped: Dict[str, List[Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]]] = {}
        for triplet in usable_pairs:
            key = str(triplet[0].get("source_file") or triplet[0].get("instance_id"))
            grouped.setdefault(key, []).append(triplet)
        chosen_key = rng.choice(sorted(grouped))
        instance, attachment_info, brs = rng.choice(grouped[chosen_key])

        candidates = self.realizer.realize(
            brs, self.operation_vocab, self.n_surface_candidates
        )
        if not candidates and self.allow_source_task_fallback:
            candidates = SourceTaskRealizer().realize(brs, self.operation_vocab, 1)
        if not candidates:
            raise GenerationFailure("language_realization_failed")

        validations = [
            self.validator.validate(q, brs, model, instance)
            for q in candidates
            if str(q).strip()
        ]
        accepted = [v for v in validations if v.accepted]
        if not accepted and self.allow_source_task_fallback:
            grounding = brs.get("grounding", {})
            source_task = str(grounding.get("source_task", "")).strip()
            if source_task and source_task not in [v.query for v in validations]:
                v = self.validator.validate(source_task, brs, model, instance)
                validations.append(v)
                if v.accepted:
                    accepted.append(v)

        if not accepted:
            raise GenerationFailure(
                "all_realizations_rejected",
                {
                    "instance_id": instance.get("instance_id"),
                    "candidate_validations": [asdict(v) for v in validations],
                },
            )

        selected = min(
            accepted,
            key=lambda v: (-v.score, len(v.query), v.query),
        )

        fragment_summary = self._match_summary(match)
        return GeneratedUserTask(
            query=selected.query,
            target_path=projection.full_path,
            behavior_projection=asdict(projection),
            matched_fragment=fragment_summary,
            grounded_instance={
                "instance_id": instance.get("instance_id"),
                "schema_id": instance.get("schema_id"),
                "source": instance.get("source"),
                "source_id": instance.get("source_id"),
                "source_file": instance.get("source_file"),
                "binding_status": instance.get("binding_status"),
            },
            brs=brs,
            candidate_validations=[asdict(v) for v in validations],
            oracle={},
            meta={
                "method": "BRS-v0.4 fragment-library behavioral reconstruction",
                "brs_version": "0.4",
                "grounding_mode": self.grounding_mode,
                "signature_policy": self.signature_policy,
                "relation_score": match["relation_score"],
                "usable_grounding_count": len(usable_pairs),
                "selected_source_file": instance.get("source_file"),
                "selected_instance_id": instance.get("instance_id"),
                "external_document_artifacts": len(
                    (brs.get("grounding") or {}).get("document_artifacts", [])
                ),
                "oracle_status": "deferred_outside_BRS",
            },
        )
