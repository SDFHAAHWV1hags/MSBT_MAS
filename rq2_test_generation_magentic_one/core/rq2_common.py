#!/usr/bin/env python3
from __future__ import annotations
import gzip, json, math, random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple
from adaptive_search import Individual, Gene, OBJECTIVE_DIRECTIONS, ModelRuntime, CampaignArchive, generate_complete_path, evaluate
from checkpoint_io import load_checkpoint, restore_archive


def individual_dict(x: Individual) -> Dict[str,Any]:
    return {
      'path':[g.to_dict() for g in x.genes], 'path_length':len(x.genes),
      'objectives':list(map(float,x.objectives))
    }


def nondominated(pop: Sequence[Individual]) -> List[Individual]:
    out=[]
    for i,a in enumerate(pop):
        dominated=False
        for j,b in enumerate(pop):
            if i==j: continue
            better_or_equal=True; strictly=False
            for va,vb,d in zip(a.objectives,b.objectives,OBJECTIVE_DIRECTIONS):
                aa,bb=d*va,d*vb
                if bb < aa: better_or_equal=False; break
                if bb > aa: strictly=True
            if better_or_equal and strictly:
                dominated=True; break
        if not dominated: out.append(a.clone())
    seen=set(); uniq=[]
    for x in out:
        k=(tuple(round(v,12) for v in x.objectives),tuple(g.key for g in x.genes))
        if k not in seen: seen.add(k);uniq.append(x)
    return uniq


def load_replay_context(model_path: Path, checkpoint_path: Path):
    model=json.loads(Path(model_path).read_text(encoding='utf-8'))
    ck=load_checkpoint(checkpoint_path)
    if ck.get('model_id')!=model.get('model_id'):
        raise ValueError(f'Checkpoint model {ck.get("model_id")} != {model.get("model_id")}')
    return model, ModelRuntime(model), restore_archive(ck), ck


def make_initial_population(runtime: ModelRuntime, archive: CampaignArchive, *, seed:int, pop_size:int,
                            max_path_len:int,path_gen_retries:int,terminal_template_weight:float,
                            k_neighbors:int,alpha:float,beta:float) -> Tuple[random.Random,List[Individual]]:
    rng=random.Random(seed)
    pop=[generate_complete_path(runtime,rng,max_path_len,terminal_template_weight,path_gen_retries) for _ in range(pop_size)]
    for x in pop:evaluate(x,runtime,archive,k_neighbors,alpha,beta)
    return rng,pop


def save_json_gz(path: Path, data: Dict[str,Any]):
    path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt',encoding='utf-8') as f: json.dump(data,f,ensure_ascii=False)


def load_json_any(path: Path) -> Dict[str,Any]:
    if str(path).endswith('.gz'):
        with gzip.open(path,'rt',encoding='utf-8') as f:return json.load(f)
    return json.loads(path.read_text(encoding='utf-8'))


def normalize_points(points: Sequence[Sequence[float]], bounds: Sequence[Tuple[float,float]], directions=OBJECTIVE_DIRECTIONS) -> List[Tuple[float,...]]:
    out=[]
    for p in points:
        q=[]
        for v,(lo,hi),d in zip(p,bounds,directions):
            if hi<=lo: z=0.0
            elif d>0: z=(hi-v)/(hi-lo)
            else: z=(v-lo)/(hi-lo)
            q.append(min(1.0,max(0.0,float(z))))
        out.append(tuple(q))
    return out


def empirical_bounds(point_sets: Iterable[Sequence[Sequence[float]]]) -> List[Tuple[float,float]]:
    pts=[tuple(map(float,p)) for s in point_sets for p in s]
    if not pts: raise ValueError('No objective points')
    return [(min(p[j] for p in pts),max(p[j] for p in pts)) for j in range(len(pts[0]))]


def nd_min(points: Sequence[Sequence[float]]) -> List[Tuple[float,...]]:
    pts=list(dict.fromkeys(tuple(map(float,p)) for p in points))
    out=[]
    for i,a in enumerate(pts):
        dom=False
        for j,b in enumerate(pts):
            if i==j:continue
            if all(b[k]<=a[k]+1e-15 for k in range(len(a))) and any(b[k]<a[k]-1e-15 for k in range(len(a))):
                dom=True;break
        if not dom:out.append(a)
    return out


def hypervolume_min(points: Sequence[Sequence[float]], reference: Sequence[float]) -> float:
    ref=tuple(map(float,reference));d=len(ref)
    pts=[tuple(map(float,p)) for p in points if len(p)==d and all(float(p[j])<=ref[j]+1e-15 for j in range(d))]
    pts=nd_min(pts)
    if not pts:return 0.0
    def rec(ps: List[Tuple[float,...]], r: Tuple[float,...]) -> float:
        dim=len(r)
        if not ps:return 0.0
        if dim==1:return max(0.0,r[0]-min(p[0] for p in ps))
        xs=sorted(set(p[0] for p in ps if p[0]<r[0]))
        if not xs:return 0.0
        xs.append(r[0]);active=[];total=0.0
        byx={x:[] for x in xs[:-1]}
        for p in ps:
            if p[0]<r[0]:byx[p[0]].append(p[1:])
        for i,x in enumerate(xs[:-1]):
            active.extend(byx[x]);width=xs[i+1]-x
            if width>0: total += width*rec(nd_min(active),r[1:])
        return total
    return rec(pts,ref)


def igd_plus(approx: Sequence[Sequence[float]], reference_front: Sequence[Sequence[float]]) -> float:
    A=[tuple(map(float,p)) for p in approx];R=[tuple(map(float,p)) for p in reference_front]
    if not A or not R:return float('inf')
    def dplus(a,r):
        return math.sqrt(sum(max(a[j]-r[j],0.0)**2 for j in range(len(r))))
    return sum(min(dplus(a,r) for a in A) for r in R)/len(R)
