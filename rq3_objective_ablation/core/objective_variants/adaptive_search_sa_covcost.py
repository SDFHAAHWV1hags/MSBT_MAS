#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import math
import random
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_POP_SIZE = 100
DEFAULT_GENERATIONS = 200
DEFAULT_CROSSOVER_RATE = 0.80
DEFAULT_MUTATION_RATE = 0.35
DEFAULT_MAX_PATH_LEN = 30
DEFAULT_PATH_GEN_RETRIES = 30
DEFAULT_TERMINAL_TEMPLATE_WEIGHT = 0.20
DEFAULT_K_NEIGHBORS = 5
DEFAULT_BETA_ALPHA = 1.0
DEFAULT_BETA_BETA = 1.0
DEFAULT_RANDOM_SEED = 7

OBJECTIVE_DIRECTIONS = (1, 1, 1, -1)
ACTIVE_OBJECTIVE_INDICES = (0, 3)
OBJECTIVE_NAMES = ("coverage_gain", "behavioral_diversity", "neutral", "execution_cost")


@dataclass(frozen=True)
class Gene:
    rule_id: str
    event: str
    params: Tuple[Tuple[str, str], ...] = tuple()

    @property
    def key(self) -> str:
        return f"{self.rule_id}::{self.event}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_id": self.rule_id,
            "event": self.event,
            "params": dict(self.params),
            "key": self.key,
        }


@dataclass
class Individual:
    genes: List[Gene]
    objectives: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    rank: int = 10**9
    crowding: float = 0.0
    state_trace: List[Dict[str, Any]] = field(default_factory=list)

    def clone(self) -> "Individual":
        return copy.deepcopy(self)


@dataclass
class TransitionStats:
    opportunities: int = 0
    successes: int = 0


@dataclass
class ExecutedTest:
    target_path: List[Gene]
    natural_language_test: str
    actual_path: List[Gene]
    outcome: str
    execution_log: List[Dict[str, Any]]


@dataclass
class CampaignArchive:
    executed_tests: List[ExecutedTest] = field(default_factory=list)
    covered_features: set = field(default_factory=set)
    transition_stats: Dict[str, TransitionStats] = field(default_factory=dict)

    @property
    def actual_paths(self) -> List[List[Gene]]:
        return [x.actual_path for x in self.executed_tests]


class ModelRuntime:

    def __init__(self, model: Dict[str, Any]):
        self.model = model
        self.family = model.get("family", "")
        self.structure = model["structure"]
        self.rules = {r["id"]: r for r in model["transition_rules"]}
        self.agents = list(self.structure.get("agents", []))
        self.tools = list(self.structure.get("tools", []))
        self.bindings = [(x["agent"], x["tool"]) for x in self.structure.get("agent_tool_bindings", [])]
        self.initial_state = copy.deepcopy(model["state_model"]["initial_state"])
        self.orchestrator = self.structure.get("agent_coordination", {}).get("selector")
        self.is_captain = self.family.startswith("CaptainAgent")
        self.is_magentic = "Magentic-One" in self.family or "MagenticOne" in model.get("model_id", "")

    @staticmethod
    def _sig(state: Dict[str, Any]) -> str:
        return json.dumps(state, sort_keys=True, ensure_ascii=False)

    def is_terminal(self, state: Dict[str, Any]) -> bool:
        return state.get("activeEntity") == "END"

    @staticmethod
    def _inc_stall(v: Any) -> str:
        if str(v) == "3+":
            return "3+"
        try:
            n = int(v)
        except Exception:
            n = 0
        return "3+" if n >= 2 else str(n + 1)

    def enabled_genes(self, state: Dict[str, Any]) -> List[Gene]:
        if self.is_terminal(state):
            return []
        if self.is_captain:
            return self._enabled_captain(state)
        if self.is_magentic:
            return self._enabled_magentic(state)
        raise NotImplementedError(f"Unsupported model family: {self.family}")

    def _enabled_captain(self, s: Dict[str, Any]) -> List[Gene]:
        ae = s["activeEntity"]
        budget = int(s.get("budget", 0))
        out: List[Gene] = []
        if ae == "START":
            if budget > 0:
                for a in self.agents:
                    out.append(Gene("CA-R1", f"SELECT_INITIAL({a})", (("agent", a),)))
            return out

        if ae in self.agents:
            if budget > 0:
                for a in self.agents:
                    out.append(Gene("CA-R2", f"SELECT_NEXT({a})", (("agent", a),)))
            for a, t in self.bindings:
                if a == ae:
                    out.append(Gene("CA-R3", f"TOOL_CALL({a},{t})", (("agent", a), ("tool", t))))
            out.append(Gene("CA-R5", "TERMINATE"))

        elif ae in self.tools:
            for a, t in self.bindings:
                if t == ae:
                    out.append(Gene("CA-R4", f"TOOL_RETURN({t},{a})", (("agent", a), ("tool", t))))

        if budget == 0 and ae != "END":
            out.append(Gene("CA-R6", "BUDGET_TERMINATE"))
        return out

    def _enabled_magentic(self, s: Dict[str, Any]) -> List[Gene]:
        ae = s["activeEntity"]
        phase = s.get("phase")
        budget = int(s.get("budget", 0))
        out: List[Gene] = []

        by_prefix: Dict[str, List[Tuple[str, str]]] = {}
        for rid, r in self.rules.items():
            by_prefix.setdefault(r["event"].split("(")[0], []).append((rid, r["event"]))

        def rid_for(prefix: str, exact: Optional[str] = None) -> Optional[str]:
            arr = by_prefix.get(prefix, [])
            if not arr:
                return None
            if exact is not None:
                for rid, ev in arr:
                    if ev == exact:
                        return rid
            return arr[0][0]

        if ae == "START":
            rid = rid_for("RECEIVE", "RECEIVE")
            if rid:
                out.append(Gene(rid, "RECEIVE"))
            return out

        if ae == self.orchestrator:
            if phase in {"Planning", "Replanning"}:
                rid = rid_for("PLAN", "PLAN")
                if rid:
                    out.append(Gene(rid, "PLAN"))

            if phase == "Executing":
                targets = []
                for e in self.structure.get("agent_coordination", {}).get("edges", []):
                    if e.get("source") == self.orchestrator and e.get("target") in self.agents:
                        targets.append(e["target"])
                targets = list(dict.fromkeys(targets))
                if budget > 0:
                    rid = rid_for("DELEGATE")
                    if rid:
                        for a in targets:
                            out.append(Gene(rid, f"DELEGATE({a})", (("agent", a),)))

                for ev in ["ASSESS_COMPLETE", "ASSESS_PROGRESS", "ASSESS_STALL", "ASSESS_LOOP"]:
                    rid = rid_for(ev, ev)
                    if rid:
                        out.append(Gene(rid, ev))

                if str(s.get("stallCount")) == "3+":
                    rid = rid_for("REPLAN", "REPLAN")
                    if rid:
                        out.append(Gene(rid, "REPLAN"))
                if str(s.get("requestSatisfied")) == "True":
                    rid = rid_for("FINALIZE", "FINALIZE")
                    if rid:
                        out.append(Gene(rid, "FINALIZE"))

        elif ae in self.agents:
            for a, t in self.bindings:
                if a == ae:
                    exact = f"TOOL_CALL({a},{t})"
                    rid = rid_for("TOOL_CALL", exact)
                    if rid:
                        out.append(Gene(rid, exact, (("agent", a), ("tool", t))))
            if ae != self.orchestrator:
                can_report = any(
                    e.get("source") == ae and e.get("target") == self.orchestrator
                    for e in self.structure.get("agent_coordination", {}).get("edges", [])
                )
                rid = rid_for("REPORT")
                if can_report and rid:
                    out.append(Gene(rid, f"REPORT({ae})", (("agent", ae),)))

        elif ae in self.tools:
            for a, t in self.bindings:
                if t == ae:
                    exact = f"TOOL_RETURN({t},{a})"
                    rid = rid_for("TOOL_RETURN", exact)
                    if rid:
                        out.append(Gene(rid, exact, (("agent", a), ("tool", t))))

        if budget == 0 and ae != "END":
            rid = rid_for("BUDGET_TERMINATE", "BUDGET_TERMINATE")
            if rid:
                out.append(Gene(rid, "BUDGET_TERMINATE"))
        return out

    def apply(self, state: Dict[str, Any], gene: Gene) -> Dict[str, Any]:
        if gene.key not in {g.key for g in self.enabled_genes(state)}:
            raise ValueError(f"Illegal gene {gene.key} at {state}")
        s = copy.deepcopy(state)
        p = dict(gene.params)
        e = gene.event

        if self.is_captain:
            if gene.rule_id == "CA-R1":
                s["activeEntity"] = p["agent"]; s["budget"] -= 1
            elif gene.rule_id == "CA-R2":
                s["activeEntity"] = p["agent"]; s["budget"] -= 1
            elif gene.rule_id == "CA-R3":
                s["activeEntity"] = p["tool"]
            elif gene.rule_id == "CA-R4":
                s["activeEntity"] = p["agent"]
            elif gene.rule_id == "CA-R5":
                s["activeEntity"] = "END"; s["executionStatus"] = "Completed"
            elif gene.rule_id == "CA-R6":
                s["activeEntity"] = "END"; s["executionStatus"] = "Terminated"
            return s

        if e == "RECEIVE":
            s["activeEntity"] = self.orchestrator; s["phase"] = "Planning"
        elif e == "PLAN":
            s["phase"] = "Executing"
        elif e.startswith("DELEGATE("):
            s["activeEntity"] = p["agent"]; s["budget"] -= 1
        elif e.startswith("REPORT("):
            s["activeEntity"] = self.orchestrator
        elif e.startswith("TOOL_CALL("):
            s["activeEntity"] = p["tool"]
        elif e.startswith("TOOL_RETURN("):
            s["activeEntity"] = p["agent"]
        elif e == "ASSESS_COMPLETE":
            s["requestSatisfied"] = "True"
            if "progressStatus" in s: s["progressStatus"] = "Forward"
            if "loopStatus" in s: s["loopStatus"] = "NonLooping"
            if "stallCount" in s: s["stallCount"] = "0"
        elif e == "ASSESS_PROGRESS":
            s["requestSatisfied"] = "False"
            if "progressStatus" in s: s["progressStatus"] = "Forward"
            if "loopStatus" in s: s["loopStatus"] = "NonLooping"
            if "stallCount" in s: s["stallCount"] = "0"
        elif e == "ASSESS_STALL":
            s["requestSatisfied"] = "False"
            if "progressStatus" in s: s["progressStatus"] = "Stalled"
            if "stallCount" in s: s["stallCount"] = self._inc_stall(s["stallCount"])
        elif e == "ASSESS_LOOP":
            s["requestSatisfied"] = "False"
            if "loopStatus" in s: s["loopStatus"] = "Looping"
            if "stallCount" in s: s["stallCount"] = self._inc_stall(s["stallCount"])
        elif e == "REPLAN":
            s["phase"] = "Replanning"; s["requestSatisfied"] = "Unknown"; s["stallCount"] = "0"
            if "progressStatus" in s: s["progressStatus"] = "Unknown"
            if "loopStatus" in s: s["loopStatus"] = "Unknown"
        elif e == "FINALIZE":
            s["activeEntity"] = "END"; s["executionStatus"] = "Completed"; s["phase"] = "Finalizing"
        elif e == "BUDGET_TERMINATE":
            s["activeEntity"] = "END"; s["executionStatus"] = "Terminated"
        return s

    def simulate(self, genes: Sequence[Gene]) -> Tuple[bool, Dict[str, Any], List[Dict[str, Any]]]:
        state = copy.deepcopy(self.initial_state)
        states = [copy.deepcopy(state)]
        try:
            for g in genes:
                state = self.apply(state, g)
                states.append(copy.deepcopy(state))
            return True, state, states
        except ValueError:
            return False, state, states


def is_terminal_gene(g: Gene) -> bool:
    return g.event in {"TERMINATE", "FINALIZE", "BUDGET_TERMINATE"}


def template_first_choice(genes: Sequence[Gene], rng: random.Random, terminal_weight: float) -> Gene:
    buckets: Dict[str, List[Gene]] = {}
    for g in genes:
        buckets.setdefault(g.rule_id, []).append(g)
    rids = list(buckets)
    weights = [terminal_weight if any(is_terminal_gene(g) for g in buckets[r]) else 1.0 for r in rids]
    rid = rng.choices(rids, weights=weights, k=1)[0]
    return rng.choice(buckets[rid])


def shortest_terminating_suffix(runtime: ModelRuntime, start: Dict[str, Any], max_depth: int, rng: random.Random) -> Optional[List[Gene]]:
    if runtime.is_terminal(start):
        return []
    q = deque([(copy.deepcopy(start), [])])
    visited = {(runtime._sig(start), 0)}
    while q:
        state, path = q.popleft()
        if len(path) >= max_depth:
            continue
        enabled = list(runtime.enabled_genes(state))
        rng.shuffle(enabled)
        enabled.sort(key=lambda g: 0 if is_terminal_gene(g) else 1)
        for g in enabled:
            ns = runtime.apply(state, g)
            np = path + [g]
            if runtime.is_terminal(ns):
                return np
            key = (runtime._sig(ns), len(np))
            if key not in visited:
                visited.add(key)
                q.append((ns, np))
    return None


def generate_complete_path(
    runtime: ModelRuntime,
    rng: random.Random,
    max_path_len: int,
    terminal_template_weight: float,
    retries: int,
) -> Individual:
    for _ in range(retries):
        state = copy.deepcopy(runtime.initial_state)
        genes: List[Gene] = []
        states = [copy.deepcopy(state)]

        while not runtime.is_terminal(state) and len(genes) < max_path_len:
            remaining = max_path_len - len(genes)
            if remaining <= 3:
                suffix = shortest_terminating_suffix(runtime, state, remaining, rng)
                if suffix is not None:
                    for g in suffix:
                        genes.append(g); state = runtime.apply(state, g); states.append(copy.deepcopy(state))
                    break

            enabled = runtime.enabled_genes(state)
            if not enabled:
                break
            g = template_first_choice(enabled, rng, terminal_template_weight)
            genes.append(g)
            state = runtime.apply(state, g)
            states.append(copy.deepcopy(state))

        if runtime.is_terminal(state):
            return Individual(genes=genes, state_trace=states)

        remaining = max_path_len - len(genes)
        if remaining > 0:
            suffix = shortest_terminating_suffix(runtime, state, remaining, rng)
            if suffix is not None:
                for g in suffix:
                    genes.append(g); state = runtime.apply(state, g); states.append(copy.deepcopy(state))
                if runtime.is_terminal(state):
                    return Individual(genes=genes, state_trace=states)

    raise RuntimeError("Unable to generate a complete legal path within max_path_len; increase bound or inspect termination semantics.")


def mutate_complete_path(
    parent: Individual,
    runtime: ModelRuntime,
    rng: random.Random,
    max_path_len: int,
    terminal_template_weight: float,
    retries: int,
) -> Individual:
    if not parent.genes:
        return generate_complete_path(runtime, rng, max_path_len, terminal_template_weight, retries)

    cut = rng.randint(0, len(parent.genes) - 1)
    prefix = parent.genes[:cut]
    state = copy.deepcopy(runtime.initial_state)
    prefix_states = [copy.deepcopy(state)]
    try:
        for g in prefix:
            state = runtime.apply(state, g); prefix_states.append(copy.deepcopy(state))
    except ValueError:
        return parent.clone()

    cap = max_path_len - len(prefix)
    suffix = shortest_terminating_suffix(runtime, state, cap, rng)
    if suffix is None:
        for _ in range(retries):
            s = copy.deepcopy(state); genes: List[Gene] = []
            while not runtime.is_terminal(s) and len(genes) < cap:
                rem = cap - len(genes)
                if rem <= 3:
                    rep = shortest_terminating_suffix(runtime, s, rem, rng)
                    if rep is not None:
                        genes += rep
                        for x in rep: s = runtime.apply(s, x)
                        break
                en = runtime.enabled_genes(s)
                if not en: break
                g = template_first_choice(en, rng, terminal_template_weight)
                genes.append(g); s = runtime.apply(s, g)
            if runtime.is_terminal(s):
                suffix = genes; break
    if suffix is None:
        return parent.clone()

    child = Individual(genes=list(prefix) + list(suffix))
    ok, end, states = runtime.simulate(child.genes)
    if not ok or not runtime.is_terminal(end) or len(child.genes) > max_path_len:
        return parent.clone()
    child.state_trace = states
    return child


def crossover_complete_paths(
    a: Individual,
    b: Individual,
    runtime: ModelRuntime,
    rng: random.Random,
    max_path_len: int,
) -> Tuple[Individual, Individual]:
    ok_a, end_a, sa = runtime.simulate(a.genes)
    ok_b, end_b, sb = runtime.simulate(b.genes)
    if not (ok_a and ok_b and runtime.is_terminal(end_a) and runtime.is_terminal(end_b)):
        return a.clone(), b.clone()

    pairs = []
    for i in range(1, len(sa)):
        for j in range(1, len(sb)):
            if runtime._sig(sa[i]) == runtime._sig(sb[j]):
                pairs.append((i, j))
    if not pairs:
        return a.clone(), b.clone()

    i, j = rng.choice(pairs)
    g1 = a.genes[:i] + b.genes[j:]
    g2 = b.genes[:j] + a.genes[i:]
    if not g1 or not g2 or len(g1) > max_path_len or len(g2) > max_path_len:
        return a.clone(), b.clone()

    children = []
    for gs, fallback in [(g1, a), (g2, b)]:
        ok, end, states = runtime.simulate(gs)
        if ok and runtime.is_terminal(end):
            children.append(Individual(genes=list(gs), state_trace=states))
        else:
            children.append(fallback.clone())
    return children[0], children[1]


def normalized_levenshtein(a: Sequence[str], b: Sequence[str]) -> float:
    if not a and not b: return 0.0
    if not a or not b: return 1.0
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cost = 0 if x == y else 1
            cur.append(min(cur[-1] + 1, prev[j] + 1, prev[j-1] + cost))
        prev = cur
    return prev[-1] / max(len(a), len(b))


def coverage_features(runtime: ModelRuntime, genes: Sequence[Gene]) -> set:
    ok, _, states = runtime.simulate(genes)
    if not ok: return set()
    out = {f"T::{g.key}" for g in genes}
    out.update(f"S::{runtime._sig(s)}" for s in states[1:])
    return out


def behavioral_diversity(genes: Sequence[Gene], archive: CampaignArchive, k: int) -> float:
    if not archive.actual_paths:
        return 1.0
    seq = [g.key for g in genes]
    ds = sorted(normalized_levenshtein(seq, [g.key for g in p]) for p in archive.actual_paths)
    kk = min(k, len(ds))
    return sum(ds[:kk]) / kk


def transition_realizability(g: Gene, archive: CampaignArchive, alpha: float, beta: float) -> float:
    st = archive.transition_stats.get(g.key, TransitionStats())
    return (st.successes + alpha) / (st.opportunities + alpha + beta)


def path_realizability(genes: Sequence[Gene], archive: CampaignArchive, alpha: float, beta: float) -> float:
    if not genes: return 0.0
    ps = [transition_realizability(g, archive, alpha, beta) for g in genes]
    return math.exp(sum(math.log(max(1e-12, p)) for p in ps) / len(ps))


def evaluate(ind: Individual, runtime: ModelRuntime, archive: CampaignArchive, k: int, alpha: float, beta: float) -> None:
    ok, end, states = runtime.simulate(ind.genes)
    if not ok or not runtime.is_terminal(end):
        raise AssertionError("NSGA-II individual is not a complete legal path")
    ind.state_trace = states
    cov = float(len(coverage_features(runtime, ind.genes) - archive.covered_features))
    div = behavioral_diversity(ind.genes, archive, k)
    cost = float(len(ind.genes))
    ind.objectives = (cov, div, 0.0, cost)


def dominates(a: Individual, b: Individual) -> bool:
    better = False
    for j in ACTIVE_OBJECTIVE_INDICES:
        d = OBJECTIVE_DIRECTIONS[j]
        aa, bb = d * a.objectives[j], d * b.objectives[j]
        if aa < bb: return False
        if aa > bb: better = True
    return better


def fast_non_dominated_sort(pop: Sequence[Individual]) -> List[List[Individual]]:
    dom = {i: [] for i in range(len(pop))}
    count = [0] * len(pop)
    fronts: List[List[int]] = [[]]
    for i, p in enumerate(pop):
        for j, q in enumerate(pop):
            if i == j: continue
            if dominates(p, q): dom[i].append(j)
            elif dominates(q, p): count[i] += 1
        if count[i] == 0:
            p.rank = 0; fronts[0].append(i)
    f = 0
    while f < len(fronts) and fronts[f]:
        nxt = []
        for i in fronts[f]:
            for j in dom[i]:
                count[j] -= 1
                if count[j] == 0:
                    pop[j].rank = f + 1; nxt.append(j)
        if nxt: fronts.append(nxt)
        f += 1
    return [[pop[i] for i in fr] for fr in fronts if fr]


def assign_crowding(front: List[Individual]) -> None:
    for x in front: x.crowding = 0.0
    if len(front) <= 2:
        for x in front: x.crowding = float("inf")
        return
    for m in ACTIVE_OBJECTIVE_INDICES:
        front.sort(key=lambda x: x.objectives[m])
        front[0].crowding = front[-1].crowding = float("inf")
        lo, hi = front[0].objectives[m], front[-1].objectives[m]
        if hi == lo: continue
        for i in range(1, len(front)-1):
            if not math.isinf(front[i].crowding):
                front[i].crowding += (front[i+1].objectives[m] - front[i-1].objectives[m]) / (hi-lo)


def rank_and_crowding(pop: Sequence[Individual]) -> List[List[Individual]]:
    fronts = fast_non_dominated_sort(pop)
    for f in fronts: assign_crowding(f)
    return fronts


def tournament(a: Individual, b: Individual, rng: random.Random) -> Individual:
    if a.rank != b.rank: return a if a.rank < b.rank else b
    if a.crowding != b.crowding: return a if a.crowding > b.crowding else b
    return rng.choice([a,b])


def environmental_selection(combined: List[Individual], n: int) -> List[Individual]:
    fronts = rank_and_crowding(combined)
    out: List[Individual] = []
    for f in fronts:
        if len(out) + len(f) <= n:
            out.extend(f)
        else:
            f.sort(key=lambda x: x.crowding, reverse=True)
            out.extend(f[:n-len(out)])
            break
    rank_and_crowding(out)
    return out


def _feasible_complete_path(
    runtime: ModelRuntime,
    rng: random.Random,
    is_feasible: Optional[Callable[[Sequence[Gene]], bool]],
    max_path_len: int,
    terminal_template_weight: float,
    path_gen_retries: int,
    *,
    max_attempts: int = 500,
) -> Individual:
    ind = generate_complete_path(
        runtime, rng, max_path_len, terminal_template_weight, path_gen_retries
    )
    if is_feasible is None:
        return ind
    for _ in range(max_attempts):
        if is_feasible(ind.genes):
            return ind
        ind = generate_complete_path(
            runtime, rng, max_path_len, terminal_template_weight, path_gen_retries
        )
    raise RuntimeError(
        "feasibility_constraint_exhausted: could not draw a task-level-operation "
        f"path within {max_attempts} attempts"
    )


def _repair_or_replace_child(
    child: Individual,
    runtime: ModelRuntime,
    rng: random.Random,
    is_feasible: Optional[Callable[[Sequence[Gene]], bool]],
    max_path_len: int,
    terminal_template_weight: float,
    path_gen_retries: int,
    *,
    repair_tries: int = 5,
) -> Individual:
    if is_feasible is None or is_feasible(child.genes):
        return child
    for _ in range(repair_tries):
        repaired = mutate_complete_path(
            child, runtime, rng, max_path_len,
            terminal_template_weight, path_gen_retries,
        )
        if is_feasible(repaired.genes):
            return repaired
    return _feasible_complete_path(
        runtime, rng, is_feasible, max_path_len,
        terminal_template_weight, path_gen_retries,
    )


def nsga2_complete_path_search(
    runtime: ModelRuntime,
    archive: CampaignArchive,
    rng: random.Random,
    *,
    pop_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    max_path_len: int,
    path_gen_retries: int,
    terminal_template_weight: float,
    k_neighbors: int,
    alpha: float,
    beta: float,
    is_feasible: Optional[Callable[[Sequence[Gene]], bool]] = None,
) -> Tuple[List[Individual], List[Individual]]:
    pop = [
        _feasible_complete_path(
            runtime, rng, is_feasible, max_path_len,
            terminal_template_weight, path_gen_retries,
        )
        for _ in range(pop_size)
    ]
    for x in pop: evaluate(x, runtime, archive, k_neighbors, alpha, beta)
    rank_and_crowding(pop)

    for _ in range(generations):
        children: List[Individual] = []
        while len(children) < pop_size:
            p1 = tournament(rng.choice(pop), rng.choice(pop), rng)
            p2 = tournament(rng.choice(pop), rng.choice(pop), rng)
            if rng.random() < crossover_rate:
                c1, c2 = crossover_complete_paths(p1, p2, runtime, rng, max_path_len)
            else:
                c1, c2 = p1.clone(), p2.clone()
            if rng.random() < mutation_rate:
                c1 = mutate_complete_path(c1, runtime, rng, max_path_len, terminal_template_weight, path_gen_retries)
            if rng.random() < mutation_rate:
                c2 = mutate_complete_path(c2, runtime, rng, max_path_len, terminal_template_weight, path_gen_retries)
            c1 = _repair_or_replace_child(
                c1, runtime, rng, is_feasible, max_path_len,
                terminal_template_weight, path_gen_retries,
            )
            c2 = _repair_or_replace_child(
                c2, runtime, rng, is_feasible, max_path_len,
                terminal_template_weight, path_gen_retries,
            )
            children.extend([c1,c2])
        children = children[:pop_size]
        for x in children: evaluate(x, runtime, archive, k_neighbors, alpha, beta)
        pop = environmental_selection(pop + children, pop_size)

    fronts = rank_and_crowding(pop)
    return [x.clone() for x in fronts[0]], pop


def select_pareto_ideal_point(pareto: Sequence[Individual]) -> Individual:
    if not pareto:
        raise ValueError("Empty Pareto front")
    vals = [x.objectives for x in pareto]
    mins = [min(v[j] for v in vals) for j in range(4)]
    maxs = [max(v[j] for v in vals) for j in range(4)]

    def utility(x: Individual) -> Tuple[float, float, float, float]:
        u = []
        for j in ACTIVE_OBJECTIVE_INDICES:
            v = x.objectives[j]
            direction = OBJECTIVE_DIRECTIONS[j]
            lo, hi = mins[j], maxs[j]
            if hi <= lo:
                z = 1.0
            elif direction > 0:
                z = (v - lo) / (hi - lo)
            else:
                z = (hi - v) / (hi - lo)
            u.append(z)
        return tuple(u)

    def distance(x: Individual) -> float:
        return math.sqrt(sum((1.0 - z) ** 2 for z in utility(x)))

    return min(
        pareto,
        key=lambda x: (distance(x), len(x.genes), tuple(g.key for g in x.genes)),
    ).clone()


def select_pareto_random(pareto: Sequence[Individual], rng: random.Random) -> Individual:
    return rng.choice(list(pareto)).clone()


def concretize_path_placeholder(path: Sequence[Gene], rng: random.Random) -> str:
    templates = [
        "Gather the necessary information using the available capabilities, verify it, and provide a concise final answer.",
        "Complete a coordinated research task that requires several available capabilities and return a verified result.",
        "Solve the task through coordinated information gathering and verification before producing the final response.",
    ]
    return rng.choice(templates)


def execute_mas_placeholder(task: str, target_path: Sequence[Gene]) -> Tuple[List[Dict[str, Any]], str]:
    log = [
        {"step": i, "rule_id": g.rule_id, "event": g.event, "params": dict(g.params), "message": f"MOCK executed {g.event}"}
        for i, g in enumerate(target_path)
    ]
    return log, "MOCK_SUCCESS"


def analyze_log_placeholder(log: Sequence[Dict[str, Any]]) -> List[Gene]:
    return [
        Gene(x["rule_id"], x["event"], tuple(sorted((str(k), str(v)) for k,v in x.get("params",{}).items())))
        for x in log
    ]


def update_archive(
    archive: CampaignArchive,
    runtime: ModelRuntime,
    target: Sequence[Gene],
    actual: Sequence[Gene],
    task: str,
    outcome: str,
    log: Sequence[Dict[str, Any]],
) -> None:
    archive.covered_features |= coverage_features(runtime, actual)

    for i, tg in enumerate(target):
        if i >= len(actual): break
        st = archive.transition_stats.setdefault(tg.key, TransitionStats())
        st.opportunities += 1
        if actual[i].key == tg.key:
            st.successes += 1
        else:
            break

    archive.executed_tests.append(ExecutedTest(list(target), task, list(actual), outcome, list(log)))


def run_campaign(
    model: Dict[str, Any],
    *,
    num_tests: int,
    seed: int,
    pop_size: int,
    generations: int,
    crossover_rate: float,
    mutation_rate: float,
    max_path_len: int,
    path_gen_retries: int,
    terminal_template_weight: float,
    k_neighbors: int,
    alpha: float,
    beta: float,
) -> Dict[str, Any]:
    rng = random.Random(seed)
    runtime = ModelRuntime(model)
    archive = CampaignArchive()
    report = {
        "implementation_version":"2.0-complete-path",
        "model_id":model.get("model_id"),
        "seed":seed,
        "parameters":{
            "num_tests":num_tests,"pop_size":pop_size,"generations":generations,
            "crossover_rate":crossover_rate,"mutation_rate":mutation_rate,
            "max_path_len":max_path_len,"path_gen_retries":path_gen_retries,
            "terminal_template_weight":terminal_template_weight,"k_neighbors":k_neighbors,
            "beta_alpha":alpha,"beta_beta":beta,
        },
        "tests":[],
    }

    for t in range(1, num_tests+1):
        pareto, _ = nsga2_complete_path_search(
            runtime, archive, rng,
            pop_size=pop_size, generations=generations,
            crossover_rate=crossover_rate, mutation_rate=mutation_rate,
            max_path_len=max_path_len, path_gen_retries=path_gen_retries,
            terminal_template_weight=terminal_template_weight,
            k_neighbors=k_neighbors, alpha=alpha, beta=beta,
        )
        selected = select_pareto_ideal_point(pareto)

        task = concretize_path_placeholder(selected.genes, rng)
        raw_log, outcome = execute_mas_placeholder(task, selected.genes)
        actual = analyze_log_placeholder(raw_log)
        update_archive(archive, runtime, selected.genes, actual, task, outcome, raw_log)

        report["tests"].append({
            "test_index":t,
            "nsga2_runs_for_this_test":1,
            "pareto_front_size":len(pareto),
            "pareto_front":[{
                "path":[g.to_dict() for g in x.genes],
                "path_length":len(x.genes),
                "objectives":{
                    "coverage_gain_max":x.objectives[0],
                    "behavioral_diversity_max":x.objectives[1],
                    "neutral_compatibility_slot":x.objectives[2],
                    "execution_cost_min":x.objectives[3],
                }
            } for x in pareto],
            "selected_path":[g.to_dict() for g in selected.genes],
            "selected_path_length":len(selected.genes),
            "selected_objectives":{
                "coverage_gain_max":selected.objectives[0],
                "behavioral_diversity_max":selected.objectives[1],
                "neutral_compatibility_slot":selected.objectives[2],
                "execution_cost_min":selected.objectives[3],
            },
            "natural_language_test":task,
            "mas_execution_log":raw_log,
            "actual_path":[g.to_dict() for g in actual],
            "outcome":outcome,
            "archive_after_execution":{
                "executed_tests":len(archive.executed_tests),
                "covered_feature_count":len(archive.covered_features),
                "realizability":{
                    k:{"opportunities":v.opportunities,"successes":v.successes,
                       "posterior_mean":(v.successes+alpha)/(v.opportunities+alpha+beta)}
                    for k,v in sorted(archive.transition_stats.items())
                }
            }
        })

    report["final_summary"] = {
        "executed_tests":len(archive.executed_tests),
        "actual_path_lengths":[len(x.actual_path) for x in archive.executed_tests],
        "covered_feature_count":len(archive.covered_features),
    }
    return report


def main() -> None:
    p = argparse.ArgumentParser(description="Adaptive NSGA-II v2: one complete path per search")
    p.add_argument("model", type=Path)
    p.add_argument("--output", type=Path, default=Path("adaptive_nsga2_v2_output.json"))
    p.add_argument("--num-tests", type=int, default=3)
    p.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    p.add_argument("--pop-size", type=int, default=DEFAULT_POP_SIZE)
    p.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    p.add_argument("--crossover-rate", type=float, default=DEFAULT_CROSSOVER_RATE)
    p.add_argument("--mutation-rate", type=float, default=DEFAULT_MUTATION_RATE)
    p.add_argument("--max-path-len", type=int, default=DEFAULT_MAX_PATH_LEN)
    p.add_argument("--path-gen-retries", type=int, default=DEFAULT_PATH_GEN_RETRIES)
    p.add_argument("--terminal-template-weight", type=float, default=DEFAULT_TERMINAL_TEMPLATE_WEIGHT)
    p.add_argument("--k-neighbors", type=int, default=DEFAULT_K_NEIGHBORS)
    p.add_argument("--beta-alpha", type=float, default=DEFAULT_BETA_ALPHA)
    p.add_argument("--beta-beta", type=float, default=DEFAULT_BETA_BETA)
    a = p.parse_args()

    model = json.loads(a.model.read_text(encoding="utf-8"))
    report = run_campaign(
        model,
        num_tests=a.num_tests, seed=a.seed,
        pop_size=a.pop_size, generations=a.generations,
        crossover_rate=a.crossover_rate, mutation_rate=a.mutation_rate,
        max_path_len=a.max_path_len, path_gen_retries=a.path_gen_retries,
        terminal_template_weight=a.terminal_template_weight,
        k_neighbors=a.k_neighbors, alpha=a.beta_alpha, beta=a.beta_beta,
    )
    a.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Model: {report['model_id']}")
    print(f"Version: {report['implementation_version']}")
    print(f"Executed tests: {report['final_summary']['executed_tests']}")
    print(f"Actual path lengths: {report['final_summary']['actual_path_lengths']}")
    print(f"Output: {a.output}")


if __name__ == "__main__":
    main()
