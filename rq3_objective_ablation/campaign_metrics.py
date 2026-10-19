#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,sys
from collections import deque
from pathlib import Path
ROOT=Path(__file__).resolve().parent;sys.path.insert(0,str(ROOT/'core'))
from adaptive_search import ModelRuntime, Gene, normalized_levenshtein

def genes(xs):
 out=[]
 for x in xs or []:
  p=x.get('params',{}) or {};out.append(Gene(str(x.get('rule_id','')),str(x.get('event','')),tuple(sorted((str(k),str(v)) for k,v in p.items()))))
 return out

def universe(model,max_states=200000):
 rt=ModelRuntime(model);q=deque([rt.initial_state]);seen={rt._sig(rt.initial_state)};trans=set()
 while q:
  s=q.popleft()
  for g in rt.enabled_genes(s):
   trans.add(g.key);ns=rt.apply(s,g);sig=rt._sig(ns)
   if sig not in seen:
    seen.add(sig);q.append(ns)
    if len(seen)>max_states:raise RuntimeError('Reachable-state enumeration exceeded limit')
 return seen,trans

def main():
 p=argparse.ArgumentParser();p.add_argument('campaign',type=Path);p.add_argument('model',type=Path);p.add_argument('--output',type=Path,default=None);a=p.parse_args()
 d=json.loads(a.campaign.read_text(encoding='utf-8'));m=json.loads(a.model.read_text(encoding='utf-8'));states_u,trans_u=universe(m);tests=d.get('tests',[])
 actual_paths=[genes(t.get('actual_path',[])) for t in tests];target_paths=[genes(t.get('selected_path',[])) for t in tests]
 covered_t={g.key for pth in actual_paths for g in pth};fail_sigs=[];cum=[];seen=set()
 for t in tests:
  sig=t.get('failure_signature')
  if not sig and str(t.get('outcome','')).upper() not in {'PASS','SUCCESS','OK','MOCK_SUCCESS'}:sig='OUTCOME:'+str(t.get('outcome'))
  if sig:seen.add(sig)
  cum.append(len(seen));
  if sig:fail_sigs.append(sig)
 B=len(tests);bd=[]
 for i in range(B):
  for j in range(i+1,B):bd.append(normalized_levenshtein([g.key for g in actual_paths[i]],[g.key for g in actual_paths[j]]))
 prs=[]
 for tar,act in zip(target_paths,actual_paths):
  if tar:prs.append(1.0-normalized_levenshtein([g.key for g in tar],[g.key for g in act]))
 total_sec=0.0
 for t in tests:
  total_sec+=float(t.get('search_sec',0) or 0)+float(t.get('nlp_generation_sec',0) or 0)
  tm=t.get('timings',{}) or {};total_sec+=float(tm.get('execution_sec',tm.get('total_sec',0)) or 0)
 uf=len(set(fail_sigs));out={'model_id':m.get('model_id'),'method':d.get('method'),'tests':B,'unique_failures':uf,'failed_test_rate':sum(1 for t in tests if t.get('failure_signature'))/max(1,B),'transition_coverage':len(covered_t)/max(1,len(trans_u)),'covered_transitions':len(covered_t),'reachable_transitions':len(trans_u),'failure_discovery_auc':sum(cum)/max(1,B),'behavioral_diversity':sum(bd)/max(1,len(bd)) if bd else 0.0,'path_realizability_similarity':sum(prs)/len(prs) if prs else None,'total_recorded_time_sec':total_sec,'failure_discovery_efficiency_per_hour':uf/(total_sec/3600) if total_sec>0 else None}
 text=json.dumps(out,ensure_ascii=False,indent=2);print(text)
 if a.output:a.output.write_text(text,encoding='utf-8')
if __name__=='__main__':main()
