#!/usr/bin/env python3

from __future__ import annotations

import abc
import copy
import json
import math
import random
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple




def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def unique_keep_order(items: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    out: List[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def event_prefix(event: str) -> str:
    return event.split("(", 1)[0].strip()


def event_arguments(event: str) -> List[str]:
    m = re.search(r"\((.*)\)", event)
    if not m:
        return []
    return [x.strip() for x in m.group(1).split(",") if x.strip()]


def tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


def split_identifier(text: str) -> List[str]:
    text = text.replace("-", "_")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", text)
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", text)
    return [t for t in re.findall(r"[A-Za-z0-9]+", text.replace("_", " ")) if t]


def normalize_identifier_phrase(text: str) -> str:
    return " ".join(split_identifier(text)).lower().strip()




class LLMInterface(abc.ABC):

    @abc.abstractmethod
    def synthesize(self, spec_for_llm: Dict[str, Any], n_candidates: int) -> List[str]:
        raise NotImplementedError

    def semantic_validate(self, query: str, spec_for_llm: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        return None

    def parse_library_item(self, question: str) -> Optional[Dict[str, Any]]:
        return None


class NoExternalLLM(LLMInterface):

    def synthesize(self, spec_for_llm: Dict[str, Any], n_candidates: int) -> List[str]:
        raise RuntimeError("No external LLM is configured. Use TemplateSynthesizer for an offline smoke test.")




@dataclass
class NormalizedGene:
    rule_id: str
    event: str
    params: Dict[str, str] = field(default_factory=dict)


@dataclass
class SemanticAnchor:
    path_index: int
    event: str
    implementation_visibility: str
    semantic_inducibility: str
    capability: str
    capability_strength: str
    domain_specialty: Optional[str]
    input_type: Optional[str]
    output_type: Optional[str]
    source_entity: Optional[str] = None
    target_entity: Optional[str] = None
    constraint_templates: List[str] = field(default_factory=list)
    derived_from_agent_role: bool = False


@dataclass
class PathIntent:
    goal: str
    required_capabilities: List[str]
    hard_capabilities: List[str]
    soft_capabilities: List[str]
    capability_sequence: List[str]
    domain_requirements: List[str]
    artifact_flow: List[Dict[str, Any]]
    dependencies: List[Dict[str, str]]
    model_constraints: List[str]
    output_type: str
    anchors: List[SemanticAnchor]
    internal_events: List[str]


@dataclass
class CandidateValidation:
    query: str
    accepted: bool
    score: float
    checks: Dict[str, Any]


@dataclass
class NLPTestCase:
    query: str
    target_path: List[Dict[str, Any]]
    path_intent: Dict[str, Any]
    retrieved_task_elements: List[Dict[str, Any]]
    grounded_fact: Dict[str, Any]
    constraints: List[Dict[str, Any]]
    oracle: Dict[str, Any]
    candidate_validations: List[Dict[str, Any]]
    meta: Dict[str, Any]




def normalize_path(path: Sequence[Any]) -> List[NormalizedGene]:
    out: List[NormalizedGene] = []
    for g in path:
        if isinstance(g, dict):
            params = g.get("params", {}) or {}
            if isinstance(params, list):
                params = dict(params)
            out.append(NormalizedGene(str(g.get("rule_id", "")), str(g.get("event", "")), {str(k): str(v) for k, v in params.items()}))
            continue

        rule_id = str(getattr(g, "rule_id", ""))
        event = str(getattr(g, "event", ""))
        params_raw = getattr(g, "params", {})
        if isinstance(params_raw, tuple):
            params_raw = dict(params_raw)
        out.append(NormalizedGene(rule_id, event, {str(k): str(v) for k, v in dict(params_raw or {}).items()}))
    return out




class SynonymExpander:

    def __init__(self, synonym_file: Path, use_wordnet_if_available: bool = True):
        data = load_json(synonym_file)
        self.synonyms: Dict[str, List[str]] = data.get("terms", {})
        self.use_wordnet = use_wordnet_if_available
        self._wn = None
        if use_wordnet_if_available:
            try:
                from nltk.corpus import wordnet as wn
                _ = wn.synsets("test")
                self._wn = wn
            except Exception:
                self._wn = None

    def expand_tokens(self, text: str, max_wordnet_synonyms: int = 3) -> List[str]:
        base = tokenize(text)
        expanded = list(base)
        for t in base:
            expanded.extend(tokenize(" ".join(self.synonyms.get(t, []))))
            if self._wn is not None:
                found: List[str] = []
                for syn in self._wn.synsets(t)[:3]:
                    for lemma in syn.lemma_names():
                        lemma = lemma.replace("_", " ").lower()
                        if lemma != t and lemma not in found:
                            found.append(lemma)
                        if len(found) >= max_wordnet_synonyms:
                            break
                    if len(found) >= max_wordnet_synonyms:
                        break
                expanded.extend(tokenize(" ".join(found)))
        return expanded


class LocalTfidfRetriever:

    def __init__(self, expander: SynonymExpander):
        self.expander = expander

    def rank(self, query: str, docs: Sequence[Tuple[str, str]]) -> List[Tuple[str, float]]:
        if not docs:
            return []
        q_tokens = self.expander.expand_tokens(query)
        doc_tokens = [self.expander.expand_tokens(text) for _, text in docs]
        all_docs = [q_tokens] + doc_tokens
        n = len(all_docs)
        df: Counter[str] = Counter()
        for toks in all_docs:
            for t in set(toks):
                df[t] += 1

        def vector(tokens: List[str]) -> Dict[str, float]:
            tf = Counter(tokens)
            if not tf:
                return {}
            length = float(sum(tf.values()))
            v: Dict[str, float] = {}
            for term, cnt in tf.items():
                idf = math.log((n + 1.0) / (df[term] + 1.0)) + 1.0
                v[term] = (cnt / length) * idf
            return v

        qv = vector(q_tokens)

        def cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
            if not a or not b:
                return 0.0
            dot = sum(v * b.get(k, 0.0) for k, v in a.items())
            na = math.sqrt(sum(v * v for v in a.values()))
            nb = math.sqrt(sum(v * v for v in b.values()))
            return 0.0 if na == 0.0 or nb == 0.0 else dot / (na * nb)

        scored = []
        for (doc_id, _), toks in zip(docs, doc_tokens):
            scored.append((doc_id, cosine(qv, vector(toks))))
        return sorted(scored, key=lambda x: x[1], reverse=True)




class PathSemanticAnalyzer:

    def __init__(self, mapping_file: Path):
        self.mapping = load_json(mapping_file)
        self.domain_stopwords = set(self.mapping.get("domain_stopwords", []))

    def _functional_capability(self, agent: str, model: Dict[str, Any]) -> Optional[str]:
        name_text = normalize_identifier_phrase(agent)
        ad = (model.get("agent_definitions") or {}).get(agent, {})
        role_text = ""
        if isinstance(ad, dict):
            role_text = str(ad.get("role", ""))

        for text in [name_text, role_text.lower()]:
            if not text:
                continue
            for rule in self.mapping.get("agent_role_keyword_rules", []):
                if any(k.lower().replace("_", " ") in text for k in rule.get("keywords", [])):
                    return rule["capability"]
        return None

    def _domain_specialty(self, agent: str) -> Optional[str]:
        toks = [t.lower() for t in split_identifier(agent)]
        kept = [t for t in toks if t not in self.domain_stopwords and not t.isdigit()]
        kept = [t for t in kept if t not in {"llm", "ai"}]
        if not kept:
            return None
        return " ".join(kept)

    def _agent_profile(self, agent: str, model: Dict[str, Any]) -> Dict[str, Optional[str]]:
        return {
            "functional_capability": self._functional_capability(agent, model),
            "domain_specialty": self._domain_specialty(agent),
        }

    @staticmethod
    def _source_target(g: NormalizedGene) -> Tuple[Optional[str], Optional[str]]:
        args = event_arguments(g.event)
        prefix = event_prefix(g.event)
        if prefix in {"DELEGATE", "SELECT_INITIAL", "SELECT_NEXT"}:
            target = g.params.get("agent") or (args[0] if args else None)
            return None, target
        if prefix == "REPORT":
            source = g.params.get("agent") or (args[0] if args else None)
            return source, None
        if prefix == "TOOL_CALL":
            if len(args) >= 2:
                return args[0], args[1]
            return g.params.get("agent"), g.params.get("tool")
        if prefix == "TOOL_RETURN":
            if len(args) >= 2:
                return args[0], args[1]
            return g.params.get("tool"), g.params.get("agent")
        return None, None

    def _match_rule(self, g: NormalizedGene) -> Dict[str, Any]:
        prefix = event_prefix(g.event)
        args = event_arguments(g.event)
        for rule in self.mapping.get("event_rules", []):
            if rule.get("event_prefix") != prefix:
                continue
            required_tool = rule.get("tool")
            if required_tool:
                tool = g.params.get("tool")
                if not tool and prefix == "TOOL_CALL" and len(args) >= 2:
                    tool = args[1]
                if tool != required_tool:
                    continue
            return rule
        return {
            "event_prefix": prefix,
            "implementation_visibility": "internal",
            "semantic_inducibility": "none",
            "capability": "unknown_internal_behavior",
            "input_type": None,
            "output_type": None,
        }

    @staticmethod
    def _ordered_dependency_instances(capability_sequence: Sequence[str], dep_rules: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
        dependencies: List[Dict[str, str]] = []
        for dr in dep_rules:
            before, after = dr["before"], dr["after"]
            for i, c1 in enumerate(capability_sequence):
                if c1 != before:
                    continue
                for j in range(i + 1, len(capability_sequence)):
                    if capability_sequence[j] == after:
                        item = copy.deepcopy(dr)
                        item["before_index"] = str(i)
                        item["after_index"] = str(j)
                        dependencies.append(item)
                        break
        seen = set()
        out = []
        for d in dependencies:
            key = (d.get("bridge"), d.get("before_index"), d.get("after_index"))
            if key not in seen:
                seen.add(key)
                out.append(d)
        return out

    def analyze(self, path: Sequence[Any], model: Dict[str, Any]) -> PathIntent:
        genes = normalize_path(path)
        anchors: List[SemanticAnchor] = []
        internal_events: List[str] = []
        hard_caps_seq: List[str] = []
        soft_caps_seq: List[str] = []
        cap_sequence: List[str] = []
        domains: List[str] = []
        constraints: List[str] = []
        artifact_flow: List[Dict[str, Any]] = []

        for i, g in enumerate(genes):
            rule = self._match_rule(g)
            source, target = self._source_target(g)
            cap = rule.get("capability", "unknown")
            inducibility = rule.get("semantic_inducibility", "none")
            domain = None
            derived = False

            if rule.get("derive_capability_from_agent"):
                agent = g.params.get("agent") or target
                if agent:
                    prof = self._agent_profile(agent, model)
                    if prof.get("functional_capability"):
                        cap = str(prof["functional_capability"])
                        derived = True
                    domain = prof.get("domain_specialty")

            if rule.get("derive_domain_from_agent") and not domain:
                agent = g.params.get("agent") or target
                if agent:
                    domain = self._domain_specialty(agent)

            if event_prefix(g.event) == "TOOL_CALL" and source:
                source_domain = self._domain_specialty(source)
                if source_domain:
                    domain = domain or source_domain

            strength = "hard" if inducibility == "strong" else ("soft" if inducibility == "soft" else "none")
            anchor = SemanticAnchor(
                path_index=i,
                event=g.event,
                implementation_visibility=rule.get("implementation_visibility", "internal"),
                semantic_inducibility=inducibility,
                capability=cap,
                capability_strength=strength,
                domain_specialty=domain,
                input_type=rule.get("input_type"),
                output_type=rule.get("output_type"),
                source_entity=source,
                target_entity=target,
                constraint_templates=list(rule.get("constraint_templates", [])),
                derived_from_agent_role=derived,
            )
            anchors.append(anchor)
            constraints.extend(anchor.constraint_templates)

            if anchor.implementation_visibility == "internal":
                internal_events.append(g.event)

            if strength in {"hard", "soft"} and cap not in {"agent_role_selection", "unknown", "unknown_internal_behavior"}:
                cap_sequence.append(cap)
                if strength == "hard":
                    hard_caps_seq.append(cap)
                else:
                    soft_caps_seq.append(cap)
            if domain:
                domains.append(domain)

            if anchor.input_type or anchor.output_type:
                artifact_flow.append({
                    "path_index": i,
                    "event": g.event,
                    "input_type": anchor.input_type,
                    "output_type": anchor.output_type,
                    "capability": cap,
                    "strength": strength,
                })

        hard_caps = unique_keep_order(hard_caps_seq)
        soft_caps = [c for c in unique_keep_order(soft_caps_seq) if c not in hard_caps]
        required = unique_keep_order(hard_caps + soft_caps)
        domain_requirements = unique_keep_order(domains)
        dependencies = self._ordered_dependency_instances(cap_sequence, self.mapping.get("dependency_rules", []))
        goal, output_type = self._infer_goal(required)

        return PathIntent(
            goal=goal,
            required_capabilities=required,
            hard_capabilities=hard_caps,
            soft_capabilities=soft_caps,
            capability_sequence=cap_sequence,
            domain_requirements=domain_requirements,
            artifact_flow=artifact_flow,
            dependencies=dependencies,
            model_constraints=unique_keep_order(constraints),
            output_type=output_type,
            anchors=anchors,
            internal_events=internal_events,
        )

    @staticmethod
    def _infer_goal(capabilities: List[str]) -> Tuple[str, str]:
        c = set(capabilities)
        if "verification" in c:
            return "fact_verification", "verified_fact"
        if "code_generation" in c and "executable_computation" in c:
            return "code_assisted_computation", "derived_result"
        if "executable_computation" in c and "document_inspection" in c:
            return "document_based_computation", "numeric_or_derived_result"
        if "web_retrieval" in c and "document_inspection" in c:
            return "grounded_fact_retrieval", "short_fact"
        if "data_extraction" in c and "data_analysis" in c:
            return "data_analysis", "derived_result"
        if "document_inspection" in c:
            return "document_fact_retrieval", "short_fact"
        if "web_retrieval" in c:
            return "web_fact_retrieval", "short_fact"
        if "data_analysis" in c:
            return "data_analysis", "derived_result"
        if "code_generation" in c:
            return "code_generation", "code_or_result"
        if "executable_computation" in c:
            return "computation", "numeric_or_derived_result"
        return "general_task", "short_answer"




class LocalLibraries:
    def __init__(self, library_dir: Path, config: Dict[str, Any]):
        self.library_dir = library_dir
        self.config = config
        self.task_elements = load_json(library_dir / "task_elements.json").get("elements", [])
        self.fact_records = load_json(library_dir / "entity_fact_catalog.json").get("records", [])
        self.constraints = load_json(library_dir / "constraint_templates.json")
        self.oracle_rules = load_json(library_dir / "oracle_rules.json")
        self.expander = SynonymExpander(
            library_dir / "synonyms.json",
            use_wordnet_if_available=bool(config.get("use_wordnet_if_available", True)),
        )
        self.retriever = LocalTfidfRetriever(self.expander)

    @staticmethod
    def _intent_text(intent: PathIntent) -> str:
        deps = " ".join(d.get("description", "") for d in intent.dependencies)
        domains = " ".join(intent.domain_requirements)
        seq = " ".join(intent.capability_sequence)
        return " ".join([intent.goal, seq, domains, deps, intent.output_type])

    def retrieve_task_elements(self, intent: PathIntent, top_k: int) -> List[Dict[str, Any]]:
        hard_remaining = set(intent.hard_capabilities)
        soft_remaining = set(intent.soft_capabilities)
        selected: List[Dict[str, Any]] = []
        candidates = [copy.deepcopy(e) for e in self.task_elements]
        q = self._intent_text(intent)
        docs = [(e["id"], " ".join([e.get("intent", ""), e.get("text", ""), " ".join(e.get("capabilities", [])), json.dumps(e.get("slots", {}))])) for e in candidates]
        semantic = dict(self.retriever.rank(q, docs))
        cfg = self.config.get("retrieval", {})
        wh = float(cfg.get("hard_capability_weight", 2.0))
        ws = float(cfg.get("soft_capability_weight", 1.0))
        wsem = float(cfg.get("semantic_weight", 0.30))

        while (hard_remaining or soft_remaining) and len(selected) < top_k:
            best = None
            best_score = -1.0
            for e in candidates:
                if any(x.get("id") == e.get("id") for x in selected):
                    continue
                ec = set(e.get("capabilities", []))
                new_h = hard_remaining & ec
                new_s = soft_remaining & ec
                if not new_h and not new_s:
                    continue
                score = wh * len(new_h) + ws * len(new_s) + wsem * semantic.get(e["id"], 0.0)
                if score > best_score:
                    best, best_score = e, score
            if best is None:
                break
            ee = copy.deepcopy(best)
            ee["retrieval_score"] = round(best_score, 6)
            ee["new_hard_capabilities"] = sorted(hard_remaining & set(best.get("capabilities", [])))
            ee["new_soft_capabilities"] = sorted(soft_remaining & set(best.get("capabilities", [])))
            selected.append(ee)
            hard_remaining -= set(best.get("capabilities", []))
            soft_remaining -= set(best.get("capabilities", []))

        if not selected and candidates:
            ranked = sorted(candidates, key=lambda e: semantic.get(e["id"], 0.0), reverse=True)
            ee = copy.deepcopy(ranked[0])
            ee["retrieval_score"] = round(semantic.get(ee["id"], 0.0), 6)
            selected.append(ee)

        if hard_remaining:
            raise RuntimeError(f"Task Element Library cannot cover hard capabilities: {sorted(hard_remaining)}")
        return selected

    @staticmethod
    def _domain_similarity(intent_domains: Sequence[str], record_domains: Sequence[str]) -> float:
        if not intent_domains:
            return 1.0
        a = set(tokenize(" ".join(intent_domains)))
        b = set(tokenize(" ".join(record_domains)))
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def retrieve_grounded_fact(self, intent: PathIntent, elements: Sequence[Dict[str, Any]], top_k: int, rng: random.Random) -> Dict[str, Any]:
        hard = set(intent.hard_capabilities)
        soft = set(intent.soft_capabilities)
        cfg = self.config.get("retrieval", {})
        require_all_hard = bool(cfg.get("require_all_hard_capabilities_in_fact", True))
        domain_w = float(cfg.get("domain_weight", 0.35))
        sem_w = float(cfg.get("semantic_weight", 0.30))

        def _score(r: Dict[str, Any]) -> Optional[Dict[str, Any]]:
            rc = set(r.get("supported_capabilities", []))
            if require_all_hard and not hard.issubset(rc):
                return None
            hard_cov = 1.0 if not hard else len(hard & rc) / len(hard)
            soft_cov = 1.0 if not soft else len(soft & rc) / len(soft)
            rr = copy.deepcopy(r)
            rr["hard_capability_coverage"] = hard_cov
            rr["soft_capability_coverage"] = soft_cov
            rr["domain_similarity"] = self._domain_similarity(
                intent.domain_requirements, r.get("domain_specialties", []))
            return rr

        def _collect(records: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
            return [c for c in (_score(r) for r in records) if c is not None]

        candidates = _collect([r for r in self.fact_records if not r.get("is_fixture")])
        if not candidates:
            candidates = _collect([r for r in self.fact_records if r.get("is_fixture")])

        if not candidates:
            raise RuntimeError(
                "No grounded fact record supports all hard capabilities "
                f"{sorted(hard)}. Extend the local catalog or revise the path semantics."
            )

        element_text = " ".join(e.get("text", "") for e in elements)
        q = self._intent_text(intent) + " " + element_text
        docs = [(r["id"], " ".join([
            r.get("intent", ""), r.get("source_text", ""), " ".join(r.get("tags", [])),
            " ".join(r.get("domain_specialties", [])), json.dumps(r.get("slots", {}))
        ])) for r in candidates]
        sim = dict(self.retriever.rank(q, docs))

        for r in candidates:
            r["retrieval_score"] = (
                0.35 * r["soft_capability_coverage"]
                + domain_w * r["domain_similarity"]
                + sem_w * sim.get(r["id"], 0.0)
            )
        candidates.sort(key=lambda x: x["retrieval_score"], reverse=True)
        shortlist = candidates[: max(1, top_k)]
        best = shortlist[0]["retrieval_score"]
        near_best = [r for r in shortlist if best - r["retrieval_score"] <= 0.03]
        return copy.deepcopy(rng.choice(near_best or shortlist))




class ConstraintAssembler:
    def __init__(self, libraries: LocalLibraries):
        self.libs = libraries

    def assemble(self, intent: PathIntent, elements: Sequence[Dict[str, Any]], fact: Dict[str, Any]) -> Dict[str, Any]:
        requested: List[Tuple[str, str, str]] = []
        for cid in intent.model_constraints:
            requested.append((cid, "hard", "model"))
        for cap in intent.hard_capabilities:
            for cid in self.libs.constraints.get("capability_to_constraints", {}).get(cap, []):
                requested.append((cid, "hard", f"hard_capability:{cap}"))
        for cap in intent.soft_capabilities:
            for cid in self.libs.constraints.get("capability_to_constraints", {}).get(cap, []):
                requested.append((cid, "soft", f"soft_capability:{cap}"))
        for e in elements:
            for cid in e.get("constraints", []):
                strength = "hard" if set(e.get("capabilities", [])) & set(intent.hard_capabilities) else "soft"
                requested.append((cid, strength, f"task_element:{e.get('id')}"))
        for cid in self.libs.constraints.get("always", []):
            requested.append((cid, "hard" if cid in {"no_internal_implementation_leakage", "use_only_grounded_entities"} else "soft", "global"))

        templates = self.libs.constraints.get("templates", {})
        constraints: List[Dict[str, Any]] = []
        merged: Dict[str, Dict[str, Any]] = {}
        for cid, strength, source in requested:
            if cid not in templates:
                continue
            if cid not in merged:
                item = copy.deepcopy(templates[cid])
                item.update({"id": cid, "strength": strength, "sources": [source]})
                merged[cid] = item
            else:
                if strength == "hard":
                    merged[cid]["strength"] = "hard"
                merged[cid].setdefault("sources", []).append(source)
        constraints = list(merged.values())

        oracle = self._derive_oracle(fact)
        spec = {
            "goal": intent.goal,
            "required_capabilities": list(intent.required_capabilities),
            "hard_capabilities": list(intent.hard_capabilities),
            "soft_capabilities": list(intent.soft_capabilities),
            "capability_sequence": list(intent.capability_sequence),
            "domain_requirements": list(intent.domain_requirements),
            "artifact_flow": copy.deepcopy(intent.artifact_flow),
            "dependencies": copy.deepcopy(intent.dependencies),
            "grounded_slots": copy.deepcopy(fact.get("slots", {})),
            "grounded_domains": copy.deepcopy(fact.get("domain_specialties", [])),
            "constraints": constraints,
            "expected_output_type": fact.get("output_type", intent.output_type),
            "oracle": oracle,
            "grounded_fact_id": fact.get("id"),
            "source_provenance": (fact.get("oracle") or {}).get("evidence", {}),
        }
        return spec

    def _derive_oracle(self, fact: Dict[str, Any]) -> Dict[str, Any]:
        oracle = copy.deepcopy(fact.get("oracle") or {})
        if oracle.get("expected_value") is None:
            return {"available": False, "reason": "No grounded expected value in catalog."}
        answer_type = oracle.get("answer_type") or fact.get("output_type") or "short_text"
        rule = self.libs.oracle_rules.get("rules", {}).get(answer_type, {})
        oracle.setdefault("normalizer", rule.get("normalizer", "strip_casefold"))
        oracle.setdefault("comparator", rule.get("comparator", "normalized_exact"))
        oracle["available"] = True
        return oracle


def public_spec_for_llm(spec: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(spec)
    oracle = out.pop("oracle", None)
    if oracle:
        slots = spec.get("grounded_slots") or {}
        required_format = (str(slots.get("answer_format") or "").strip()
                           or str(spec.get("expected_output_type") or "").strip()
                           or "short answer")
        out["oracle_requirement"] = {
            "answer_type": oracle.get("answer_type"),
            "required_format": required_format,
            "note": ("The task must state this output format explicitly and require the answer on "
                     "its own, with no explanation, no extra prose and no Markdown decoration. "
                     "Do not reveal any expected answer value.")
        }
    return out




class TemplateSynthesizer:

    def synthesize(self, spec_for_llm: Dict[str, Any], n_candidates: int) -> List[str]:
        slots = spec_for_llm.get("grounded_slots", {})
        hard = set(spec_for_llm.get("hard_capabilities", []))
        soft = set(spec_for_llm.get("soft_capabilities", []))
        caps = hard | soft
        domains = [str(x) for x in (spec_for_llm.get("domain_requirements", []) or spec_for_llm.get("grounded_domains", [])) if str(x).strip().lower() not in {"", "general"}]
        domain_phrase = str(domains[0]) if domains else ""
        seed = str(slots.get("task_seed", "")).strip()
        requested = str(slots.get("requested_fact", "the requested result")).strip()
        answer_format = str(slots.get("answer_format", spec_for_llm.get("expected_output_type", "short answer"))).strip()

        if slots.get("document_title"):
            artifact = str(slots["document_title"])
        elif slots.get("topic"):
            artifact = f"the document concerning {slots['topic']}"
        elif slots.get("source_one") and slots.get("source_two"):
            artifact = f"{slots['source_one']} and {slots['source_two']}"
        else:
            artifact = str(slots.get("artifact", "the relevant source"))

        content = seed if seed else f"determine {requested}"
        prefix_parts: List[str] = []
        if "web_retrieval" in caps:
            prefix_parts.append("use publicly accessible online sources")
        if "document_inspection" in caps:
            prefix_parts.append(f"inspect the internal content of {artifact}")
        if "data_extraction" in caps:
            prefix_parts.append("extract the specific information or structured data needed")
        if "data_analysis" in caps:
            prefix_parts.append("analyze or compare the extracted data")
        if "code_generation" in caps:
            prefix_parts.append("produce code needed for the task")
        if "executable_computation" in caps:
            prefix_parts.append("perform the necessary executable calculation rather than relying on direct lookup")
        if "verification" in caps:
            prefix_parts.append("confirm the candidate result using independent evidence")

        process = ", then ".join(prefix_parts) if prefix_parts else "use the available evidence"
        domain_intro = f"For a task about {domain_phrase}, " if domain_phrase else ""
        bases = [
            f"{domain_intro}{process} to answer this grounded task: {content} Return the result as {answer_format}.",
            f"{domain_intro}complete the following grounded task: {content} In doing so, {process}. Return {answer_format}.",
            f"{domain_intro}{content} Your solution should {process}. Provide the final response as {answer_format}."
        ]
        return [bases[i % len(bases)] for i in range(n_candidates)]



_SEED_LITERAL_RE = re.compile(r"\[[^\[\]]{5,}\]|https?://[^\s\"'<>)\]]+")
_SEED_LITERAL_MIN_KEY = 6

_NATURALNESS_MAX_WORDS = 200


def _literal_key(text: Any) -> str:
    return re.sub(r"\s+", "", str(text)).lower()


def must_copy_literals(slots: Dict[str, Any]) -> List[str]:
    text = str((slots or {}).get("task_seed") or "")
    out: List[str] = []
    seen: Set[str] = set()
    for match in _SEED_LITERAL_RE.findall(text):
        key = _literal_key(match)
        if len(key) >= _SEED_LITERAL_MIN_KEY and key not in seen:
            seen.add(key)
            out.append(match.strip())
    return out


class LightweightValidator:
    def __init__(self, libraries: LocalLibraries, llm: Optional[LLMInterface], config: Dict[str, Any]):
        self.libs = libraries
        self.llm = llm
        self.config = config

    @staticmethod
    def _leakage_terms(model: Dict[str, Any]) -> List[str]:
        terms = []
        structure = model.get("structure", {})
        terms.extend(structure.get("agents", []))
        terms.extend(structure.get("tools", []))
        for r in model.get("transition_rules", []):
            terms.append(str(r.get("id", "")))
        generic = {"report", "plan", "finalize", "terminate", "receive"}
        return [t for t in unique_keep_order(terms) if len(t) >= 3 and t.lower() not in generic]

    @staticmethod
    def _capability_markers(cap: str) -> List[str]:
        return {
            "web_retrieval": ["web", "online", "public", "locate", "find", "source"],
            "document_inspection": ["document", "paper", "report", "pdf", "table", "page", "inside", "content"],
            "executable_computation": ["compute", "calculate", "calculation", "derive", "executable"],
            "verification": ["verify", "confirm", "cross-check", "independent", "check"],
            "data_extraction": ["extract", "identify", "retrieve", "structured data", "specific information"],
            "data_analysis": ["analyze", "analysis", "compare", "aggregate", "count", "difference", "derive"],
            "code_generation": ["code", "program", "script", "implementation", "function"],
        }.get(cap, [])

    def _capability_coverage(self, query: str, required: Sequence[str]) -> Tuple[float, Dict[str, bool]]:
        if not required:
            return 1.0, {}
        q = query.lower()
        checks: Dict[str, bool] = {}
        for cap in required:
            markers = self._capability_markers(cap)
            checks[cap] = True if not markers else any(m in q for m in markers)
        return (sum(checks.values()) / len(checks)), checks

    @staticmethod
    def _constraint_check(query: str, cid: str) -> bool:
        q = query.lower()
        if cid == "requires_public_web_source":
            return any(x in q for x in ["web", "online", "public", "source", "locate", "find"])
        if cid == "requires_document_internal_evidence":
            return any(x in q for x in ["inside", "internal content", "document", "paper", "report", "pdf", "table", "page", "content"])
        if cid == "anti_search_snippet_shortcut":
            return any(x in q for x in ["document itself", "inside", "internal content", "inspect", "search-result snippet"])
        if cid == "requires_actual_computation":
            return any(x in q for x in ["compute", "calculate", "calculation", "derive", "executable"])
        if cid == "requires_independent_verification":
            return any(x in q for x in ["independent", "verify", "confirm", "cross-check", "check"])
        if cid == "requires_data_extraction":
            return any(x in q for x in ["extract", "identify", "specific information", "structured data"])
        if cid == "requires_data_analysis":
            return any(x in q for x in ["analyze", "analysis", "compare", "aggregate", "derive", "count"])
        if cid == "requires_code_generation":
            return any(x in q for x in ["code", "program", "script", "function", "implementation"])
        return True

    @staticmethod
    def _domain_alignment(query: str, domains: Sequence[str]) -> Tuple[float, Dict[str, bool]]:
        if not domains:
            return 1.0, {}
        q_tokens = set(tokenize(query))
        checks = {}
        for d in domains:
            dt = set(tokenize(d))
            checks[d] = bool(dt & q_tokens) if dt else True
        return sum(checks.values()) / max(1, len(checks)), checks

    def validate(self, query: str, spec: Dict[str, Any], model: Dict[str, Any]) -> CandidateValidation:
        lower = query.lower()
        leakage_terms = self._leakage_terms(model)
        leaked = [t for t in leakage_terms if t.lower() in lower]
        leakage_ok = len(leaked) == 0

        hard_score, hard_checks = self._capability_coverage(query, spec.get("hard_capabilities", []))
        soft_score, soft_checks = self._capability_coverage(query, spec.get("soft_capabilities", []))

        hard_constraint_checks: Dict[str, bool] = {}
        soft_constraint_checks: Dict[str, bool] = {}
        for c in spec.get("constraints", []):
            cid = c.get("id", "")
            ok = self._constraint_check(query, cid)
            if c.get("strength") == "hard" and c.get("type") not in {"validation", "assembly"}:
                hard_constraint_checks[cid] = ok
            elif c.get("type") not in {"validation", "assembly"}:
                soft_constraint_checks[cid] = ok
        hard_constraint_score = 1.0 if not hard_constraint_checks else sum(hard_constraint_checks.values()) / len(hard_constraint_checks)
        soft_constraint_score = 1.0 if not soft_constraint_checks else sum(soft_constraint_checks.values()) / len(soft_constraint_checks)

        oracle = spec.get("oracle", {})
        oracle_ok = bool(oracle.get("available") and oracle.get("expected_value") is not None and oracle.get("comparator"))

        slots = spec.get("grounded_slots", {})
        distinctive = []
        for key in ["document_title", "organization", "page", "topic", "source_one", "source_two", "article_author", "publisher"]:
            val = str(slots.get(key, "")).strip()
            if val and len(val) >= 3:
                distinctive.append(val)
        seed_present = bool(str(slots.get("task_seed", "")).strip())
        entity_ok = seed_present or (True if not distinctive else any(v.lower() in lower for v in distinctive))

        query_key = _literal_key(query)
        missing_literals = [x for x in must_copy_literals(slots) if _literal_key(x) not in query_key]
        literals_ok = not missing_literals

        domain_score, domain_checks = self._domain_alignment(query, spec.get("domain_requirements", []))

        words = query.split()
        naturalness = 1.0 if 8 <= len(words) <= _NATURALNESS_MAX_WORDS and "{" not in query and "}" not in query else 0.5

        llm_check = None
        if self.llm is not None and self.config.get("validation", {}).get("use_llm_semantic_judge_if_available", False):
            try:
                llm_check = self.llm.semantic_validate(query, public_spec_for_llm(spec))
            except Exception as exc:
                llm_check = {"error": str(exc)}

        vc = self.config.get("validation", {})
        score = (
            float(vc.get("hard_capability_weight", 1.5)) * hard_score
            + float(vc.get("soft_capability_weight", 0.75)) * soft_score
            + float(vc.get("constraint_coverage_weight", 1.0)) * ((hard_constraint_score + soft_constraint_score) / 2.0)
            + float(vc.get("domain_alignment_weight", 0.5)) * domain_score
            + float(vc.get("naturalness_weight", 0.25)) * naturalness
        )
        if llm_check and isinstance(llm_check, dict) and isinstance(llm_check.get("capability_match"), (int, float)):
            score += 0.25 * float(llm_check["capability_match"])

        hard_requirements = []
        if vc.get("reject_internal_leakage", True):
            hard_requirements.append(leakage_ok)
        if vc.get("require_oracle", True):
            hard_requirements.append(oracle_ok)
        if vc.get("require_grounded_entities", True):
            hard_requirements.append(entity_ok)
        if vc.get("require_seed_literals", True):
            hard_requirements.append(literals_ok)
        if vc.get("require_all_hard_capabilities", True):
            hard_requirements.append(hard_score >= 1.0)
        hard_requirements.append(hard_constraint_score >= float(vc.get("hard_constraint_threshold", 1.0)))

        return CandidateValidation(
            query=query,
            accepted=all(hard_requirements),
            score=round(score, 6),
            checks={
                "internal_leakage_ok": leakage_ok,
                "leaked_terms": leaked,
                "hard_capability_score": round(hard_score, 6),
                "hard_capability_checks": hard_checks,
                "soft_capability_score": round(soft_score, 6),
                "soft_capability_checks": soft_checks,
                "hard_constraint_score": round(hard_constraint_score, 6),
                "hard_constraint_checks": hard_constraint_checks,
                "soft_constraint_score": round(soft_constraint_score, 6),
                "soft_constraint_checks": soft_constraint_checks,
                "domain_alignment_score": round(domain_score, 6),
                "domain_checks": domain_checks,
                "oracle_available": oracle_ok,
                "grounded_entity_anchor_ok": entity_ok,
                "seed_literal_ok": literals_ok,
                "missing_seed_literals": missing_literals,
                "naturalness": naturalness,
                "llm_semantic_judge": llm_check,
            },
        )




class PathConditionedConcretizer:
    def __init__(
        self,
        root_dir: Path,
        llm: Optional[LLMInterface] = None,
        use_template_fallback: bool = True,
    ):
        self.root_dir = root_dir
        self.config = load_json(root_dir / "config.json")
        self.analyzer = PathSemanticAnalyzer(root_dir / "libraries" / "semantic_mapping.json")
        self.libs = LocalLibraries(root_dir / "libraries", self.config)
        self.assembler = ConstraintAssembler(self.libs)
        self.llm = llm
        self.template = TemplateSynthesizer() if use_template_fallback else None
        self.validator = LightweightValidator(self.libs, llm, self.config)

    def generate(
        self,
        path: Sequence[Any],
        model: Dict[str, Any],
        rng: Optional[random.Random] = None,
        n_candidates: Optional[int] = None,
    ) -> NLPTestCase:
        rng = rng or random.Random(7)
        n_candidates = int(n_candidates or self.config.get("candidate_count", 3))

        intent = self.analyzer.analyze(path, model)

        elements = self.libs.retrieve_task_elements(
            intent, int(self.config.get("task_element_top_k", 5))
        )
        fact = self.libs.retrieve_grounded_fact(
            intent, elements, int(self.config.get("fact_top_k", 3)), rng
        )

        spec = self.assembler.assemble(intent, elements, fact)
        llm_spec = public_spec_for_llm(spec)

        candidates: List[str]
        synthesis_mode: str
        if self.llm is not None:
            try:
                candidates = self.llm.synthesize(llm_spec, n_candidates)
                synthesis_mode = "external_llm"
            except Exception:
                if self.template is None:
                    raise
                candidates = self.template.synthesize(llm_spec, n_candidates)
                synthesis_mode = "template_fallback_after_llm_error"
        else:
            if self.template is None:
                raise RuntimeError("No LLM configured and template fallback disabled.")
            candidates = self.template.synthesize(llm_spec, n_candidates)
            synthesis_mode = "template_fallback"

        validations = [self.validator.validate(q, spec, model) for q in candidates]
        accepted = [v for v in validations if v.accepted]
        if not validations:
            raise RuntimeError("No natural-language candidates were generated.")
        if not accepted and self.config.get("validation", {}).get("fail_if_no_candidate_passes", True):
            raise RuntimeError("All generated natural-language candidates failed hard validation.")
        pool = accepted if accepted else validations
        selected = max(pool, key=lambda x: x.score)

        genes = normalize_path(path)
        return NLPTestCase(
            query=selected.query,
            target_path=[asdict(g) for g in genes],
            path_intent={
                "goal": intent.goal,
                "required_capabilities": intent.required_capabilities,
                "hard_capabilities": intent.hard_capabilities,
                "soft_capabilities": intent.soft_capabilities,
                "capability_sequence": intent.capability_sequence,
                "domain_requirements": intent.domain_requirements,
                "artifact_flow": intent.artifact_flow,
                "dependencies": intent.dependencies,
                "model_constraints": intent.model_constraints,
                "output_type": intent.output_type,
                "anchors": [asdict(a) for a in intent.anchors],
                "internal_events": intent.internal_events,
            },
            retrieved_task_elements=elements,
            grounded_fact=fact,
            constraints=spec["constraints"],
            oracle=spec["oracle"],
            candidate_validations=[asdict(v) for v in validations],
            meta={
                "method": "path_conditioned_semantic_concretization_v1_1_frozen",
                "synthesis_mode": synthesis_mode,
                "candidate_count": len(candidates),
                "selected_score": selected.score,
                "test_model_unchanged": True,
                "expected_answer_exposed_to_synthesizer": False,
            },
        )



ANSWER_FORMAT_PHRASES: Dict[str, str] = {
    "numeric": "numeric value",
    "integer": "numeric value",
    "date": "date",
    "json": "JSON object",
    "short_text": "short factual answer",
}

_EMPHASIS_PAIRS: Tuple[Tuple[str, str], ...] = (
    ("**", "**"), ("__", "__"), ("`", "`"), ("*", "*"), ("_", "_"),
)
_QUOTE_PAIRS: Tuple[Tuple[str, str], ...] = (
    ('"', '"'), ("'", "'"), ("\u201c", "\u201d"), ("\u2018", "\u2019"),
)
_TRAILING_PUNCT = " \t\r\n.。,，;；:：!！?？"

_NUMBER_RE = re.compile(r"[+-]?(?:\d[\d,]*\.?\d+|\d[\d,]*|\.\d+)(?:[eE][+-]?\d+)?")
_CONTAINMENT_MIN_CHARS = 12


def strip_decoration(value: Any) -> str:
    t = str(value).strip()
    changed = True
    while changed:
        changed = False
        for a, b in _EMPHASIS_PAIRS + _QUOTE_PAIRS:
            if len(t) > len(a) + len(b) and t.startswith(a) and t.endswith(b):
                t = t[len(a):-len(b)].strip()
                changed = True
    return t.rstrip(_TRAILING_PUNCT)


def _unique_number(s: str) -> Optional[float]:
    toks = _NUMBER_RE.findall(s)
    if len(toks) != 1:
        return None
    try:
        return float(toks[0].replace(",", ""))
    except ValueError:
        return None


def _normalize_number(s: str, as_int: bool) -> Any:
    val = _unique_number(s)
    if val is None:
        return s
    return int(round(val)) if as_int else val


def _normalize_answer(value: Any, normalizer: str) -> Any:
    s = strip_decoration(value)
    if normalizer in {"numeric", "integer"}:
        return _normalize_number(s, normalizer == "integer")
    if normalizer == "strip_casefold":
        return re.sub(r"\s+", " ", s).strip().casefold()
    if normalizer == "strip_whitespace":
        return re.sub(r"\s+", "", s)
    if normalizer == "set_tokens":
        return {x.strip().casefold() for x in re.split(r"[,;|]", s) if x.strip()}
    if normalizer == "iso_date":
        import datetime as _dt
        for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y", "%B %d, %Y", "%b %d, %Y"):
            try:
                return _dt.datetime.strptime(s, fmt).date().isoformat()
            except Exception:
                pass
        return s.casefold()
    if normalizer == "canonical_json":
        try:
            return json.dumps(json.loads(s), sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        except Exception:
            return s
    return s


def evaluate_answer(actual_answer: Any, oracle: Dict[str, Any]) -> Dict[str, Any]:
    if not oracle.get("available"):
        return {"verdict": "UNKNOWN", "reason": "No oracle available."}
    normalizer = oracle.get("normalizer", "strip_casefold")
    expected = _normalize_answer(oracle.get("expected_value"), normalizer)
    actual = _normalize_answer(actual_answer, normalizer)
    comp = oracle.get("comparator", "normalized_exact")
    if comp == "set_equal":
        ok = set(actual) == set(expected)
    else:
        ok = actual == expected
    if (not ok and isinstance(actual, str) and isinstance(expected, str)
            and len(expected) >= _CONTAINMENT_MIN_CHARS and expected in actual):
        ok, comp = True, f"{comp}+contains"
    return {"verdict": "PASS" if ok else "FAIL", "actual": actual, "expected": expected, "comparator": comp}




def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Path-conditioned NLP-form test-case generator")
    parser.add_argument("model", type=Path)
    parser.add_argument("path", type=Path, help="JSON file containing a list of search-generated transition genes")
    parser.add_argument("--output", type=Path, default=Path("nlp_testcase_output.json"))
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    model = load_json(args.model)
    path_data = load_json(args.path)
    if isinstance(path_data, dict) and "path" in path_data:
        path_data = path_data["path"]
    concretizer = PathConditionedConcretizer(Path(__file__).resolve().parent)
    tc = concretizer.generate(path_data, model, random.Random(args.seed))
    args.output.write_text(json.dumps(asdict(tc), ensure_ascii=False, indent=2), encoding="utf-8")
    print("Generated query:")
    print(tc.query)
    print(f"Oracle available: {tc.oracle.get('available')}")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
