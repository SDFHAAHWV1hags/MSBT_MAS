#!/usr/bin/env python3
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

class GenerationFailure(RuntimeError): pass

class FragmentMatcher:
    def __init__(self, library_file: Path, seed: int = 7):
        self.lib=json.loads(Path(library_file).read_text(encoding="utf-8"))
        self.rng=random.Random(seed)
        self.elements={}
        for x in self.lib["operation_elements"]:
            self.elements.setdefault(x["signature"][0],[]).append(x)
        self.relations=self.lib["relation_fragments"]

    def match_elements(self, target: Sequence[str]):
        candidates=[]
        for i,op in enumerate(target):
            xs=self.elements.get(op,[])
            if not xs:
                raise GenerationFailure(f"schema_gap: no operation element for position {i}: {op}")
            candidates.append(xs)
        return candidates

    def relation_occurrences(self, target: Sequence[str]):
        out=[]
        t=list(target)
        for r in self.relations:
            sig=r["signature"]; k=len(sig)
            for i in range(len(t)-k+1):
                if t[i:i+k]==sig:
                    out.append({"start":i,"end":i+k-1,"fragment":r})
        return out

    def choose(self, target: Sequence[str]) -> Dict[str,Any]:
        elem_lists=self.match_elements(target)
        chosen_elements=[self.rng.choice(xs) for xs in elem_lists]

        rels=self.relation_occurrences(target)
        chosen_relations=rels

        return {
            "target_operations":list(target),
            "operation_elements":[
                {"position":i,"fragment_id":x["fragment_id"],"metadata":x}
                for i,x in enumerate(chosen_elements)
            ],
            "relation_score":len(chosen_relations),
            "relations":[
                {"start":r["start"],"end":r["end"],
                 "fragment_id":r["fragment"]["fragment_id"],
                 "relation":r["fragment"]["relation"]}
                for r in chosen_relations
            ],
            "selection_reason":(
                "relation-backed composition preferred"
                if chosen_relations else
                "no applicable relation fragment; seeded random element-compatible selection"
            )
        }

def assemble_slot_rules(match: Dict[str,Any]) -> Dict[str,Any]:
    ops=[]
    for x in match["operation_elements"]:
        meta=x["metadata"]["operation"]
        pos=x["position"]
        local={}
        for slot in meta.get("inputs",[])+meta.get("outputs",[]):
            local[slot]=f"{slot}@{pos}"
        ops.append({
            "position":pos,"type":meta["type"],
            "inputs":[local[s] for s in meta.get("inputs",[])],
            "outputs":[local[s] for s in meta.get("outputs",[])]
        })
    unions=[]
    for r in match["relations"]:
        a,b=r["start"],r["end"]
        rel=r["relation"]
        if rel["shared_role"]=="document":
            unions.append({"left":ops[a]["outputs"][0],"right":ops[b]["inputs"][0],
                           "type":rel["shared_type"],"source":r["fragment_id"]})
        elif rel["shared_role"]=="extracted_information":
            unions.append({"left":ops[a]["outputs"][0],"right":ops[b]["inputs"][0],
                           "type":rel["shared_type"],"source":r["fragment_id"]})
    return {"operations":ops,"slot_unions":unions}

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument("library",type=Path)
    ap.add_argument("--operations",required=True,
                    help="comma-separated canonical operation sequence")
    ap.add_argument("--seed",type=int,default=7)
    a=ap.parse_args()
    target=[x.strip() for x in a.operations.split(",") if x.strip()]
    m=FragmentMatcher(a.library,a.seed).choose(target)
    m["assembly"]=assemble_slot_rules(m)
    print(json.dumps(m,ensure_ascii=False,indent=2))
