#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, random, re, sys, time
from pathlib import Path
from typing import Any, Dict, List

ROOT=Path(__file__).resolve().parent
sys.path[:0]=[str(ROOT/'core')]
from adaptive_search import CampaignArchive, ExecutedTest, ModelRuntime, coverage_features
from checkpoint_io import CheckpointRecorder,load_checkpoint,deserialize_archive,decode_random_state,sha256_json
from execution_adapter import (
    ModuleExecutor, MockExecutor, extract_llm_usage,
)
from llm_client import load_llm


def public_mas_configuration(model: Dict[str,Any]) -> Dict[str,Any]:
    s=model.get('structure',{})
    return {
        'family':model.get('family'),
        'agents':s.get('agents',[]),
        'tools':s.get('tools',[]),
        'agent_tool_bindings':s.get('agent_tool_bindings',[]),
        'agent_coordination':s.get('agent_coordination',{}),
        'agent_definitions':model.get('agent_definitions',{}),
        'static_parameters':model.get('static_parameters',{}),
    }


def history_summary(records: List[Dict[str,Any]], max_items: int=20) -> List[Dict[str,Any]]:
    out=[]
    for r in records[-max_items:]:
        out.append({
            'test_index':r['test_index'],'query':r['query'],'outcome':r['outcome'],
            'failure_signature':r.get('failure_signature'),
            'actual_behavior':[x.get('event') for x in r.get('actual_path',[])],
        })
    return out


_JSON_FENCE_RE=re.compile(r'^\s*```(?:json|JSON)?\s*(.*?)\s*```\s*$',re.DOTALL)
_PLAIN_FENCE_RE=re.compile(r'^\s*```[a-zA-Z0-9_-]*\s*(.*?)\s*```\s*$',re.DOTALL)


def parse_query(text: str):
    if not text: return None
    s=text.strip()
    m=_JSON_FENCE_RE.match(s)
    if m: s=m.group(1).strip()
    candidates=[s]
    b=re.search(r'\{.*\}',s,re.DOTALL)
    if b: candidates.append(b.group(0))
    for c in candidates:
        try:
            d=json.loads(c)
        except Exception:
            continue
        if isinstance(d,dict):
            q=d.get('query') or d.get('test_case') or d.get('task')
            if q and str(q).strip(): return str(q).strip()
    return None


def _strip_fences(text: str) -> str:
    s=(text or '').strip()
    m=_PLAIN_FENCE_RE.match(s)
    return m.group(1).strip() if m else s


def generate_query(llm, model_cfg, history, *, temperature: float, max_tokens: int,
                   json_retries: int=3) -> str:
    base_system=(
      'You generate black-box natural-language test tasks for an LLM multi-agent system. '
      'Use the supplied MAS configuration and previous test/execution summaries. Generate ONE next test task that is '
      'meaningfully different from earlier tests and may exercise different supported behaviors. '
      'Task requirements: the task must be at most 100 English words. '
      'Do not reference any file, attachment, or file path that is not actually provided to the system; '
      'if the task needs data, include a small sample directly in the task text. '
      'Design the task around the agents and tools actually configured in this system, '
      'preferring tasks that can be completed within the system’s available capabilities.'
    )
    system_json=base_system+' Return JSON: {"query":"..."}.'
    user=json.dumps({'mas_configuration':model_cfg,'previous_tests':history},ensure_ascii=False)
    messages=[{'role':'system','content':system_json},{'role':'user','content':user}]
    for attempt in range(1,json_retries+1):
        raw=llm.complete(messages,temperature=temperature,max_tokens=max_tokens,response_format_json=True)
        q=parse_query(raw)
        if q:
            if attempt>1: print(f'[c1v2] query JSON parsed on attempt {attempt}/{json_retries}',file=sys.stderr)
            return q
        print(f'[c1v2] query JSON attempt {attempt}/{json_retries} unusable (raw_len={len(raw or "")}); retrying',file=sys.stderr)
    system_plain=(base_system+' Output ONLY the task itself as a single plain-text paragraph. '
                  'No JSON, no markdown, no code fences, no quotation wrappers, no explanations or preamble.')
    raw=llm.complete([{'role':'system','content':system_plain},{'role':'user','content':user}],
                     temperature=temperature,max_tokens=max_tokens,response_format_json=False)
    q=_strip_fences(raw)
    if q:
        print('[c1v2] JSON attempts all failed; using plain-text fallback output as query',file=sys.stderr)
        return q
    print('[c1v2] WARNING: plain-text fallback also empty; executing empty query',file=sys.stderr)
    return ''


def run(args):
    model=json.loads(args.model.read_text(encoding='utf-8'))
    runtime=ModelRuntime(model); archive=CampaignArchive(); rng=random.Random(args.seed); records=[]
    llm=load_llm(args.llm_config,args.llm_id,allow_mock=args.allow_mock)
    executor=MockExecutor() if args.executor is None else ModuleExecutor(args.executor)
    cid=args.campaign_id or f'{model.get("model_id")}_directllmv2_{args.llm_id}_seed{args.seed}'
    rec=CheckpointRecorder(args.checkpoint_root,cid,model,metadata={'method':'Direct-LLM-v2','llm_id':args.llm_id,'seed':args.seed})
    if args.resume:
        latest=rec.root/'LATEST'
        if not latest.exists():raise SystemExit(f'--resume: no checkpoint found under {rec.root}')
        ck=load_checkpoint(rec.root/latest.read_text(encoding='utf-8').strip())
        if ck.get('model_hash')!=sha256_json(model):raise SystemExit('--resume: model file hash differs from the checkpointed campaign')
        start_t=int(ck['test_index'])+1;archive=deserialize_archive(ck['archive'])
        if ck.get('random_state_b64'):
            rng.setstate(decode_random_state(ck['random_state_b64']))
        else:
            rng=random.Random(args.seed+start_t);print('[resume] WARNING: checkpoint lacks rng state; reseeded (not bit-exact)',file=sys.stderr)
        for k in range(1,start_t):
            row=load_checkpoint(rec.root/f'C{k:03d}.json').get('last_round')
            if not row:raise SystemExit(f'--resume: missing completed round C{k:03d}.json; cannot reconstruct prior records')
            records.append(row)
        print(f'[resume] restored {start_t-1} completed round(s); features={len(archive.covered_features)} executed_tests={len(archive.executed_tests)}; continuing at test {start_t}',file=sys.stderr)
    else:
        rec.save(0,archive,last_round={},search_parameters={},random_state=rng.getstate());start_t=1
    for t in range(start_t,args.num_tests+1):
        t0=time.perf_counter()
        q=generate_query(llm,public_mas_configuration(model),history_summary(records,args.history_items),temperature=args.temperature,max_tokens=args.max_tokens,json_retries=args.query_retries)
        gen_sec=time.perf_counter()-t0
        er=executor.execute(q,model,{'method':'Direct-LLM','test_index':t,'seed':args.seed,'model_path':str(args.model.resolve())})
        actual=executor.analyze_path(er.raw_log,model)
        archive.covered_features |= coverage_features(runtime,actual)
        archive.executed_tests.append(ExecutedTest([],q,actual,er.outcome,er.raw_log))
        row={
          'test_index':t,'query':q,'outcome':er.outcome,'failure_signature':er.failure_signature,
          'actual_path':[g.to_dict() for g in actual],'final_answer':er.final_answer,
          'timings':{'generation_sec':gen_sec,**(er.timings or {})},
          'mas_llm_usage':extract_llm_usage(er.raw_log)
        }
        records.append(row)
        rec.save(t,archive,last_round=row,search_parameters={},random_state=rng.getstate())
    out={'method':'Direct-LLM-v2','model_id':model.get('model_id'),'llm_id':args.llm_id,'seed':args.seed,'num_tests':args.num_tests,'tests':records}
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('--llm-config',type=Path,required=True);p.add_argument('--llm-id',required=True)
    p.add_argument('--executor',type=Path,default=None);p.add_argument('--allow-mock',action='store_true');p.add_argument('--num-tests',type=int,default=30);p.add_argument('--seed',type=int,default=1)
    p.add_argument('--temperature',type=float,default=0.2);p.add_argument('--max-tokens',type=int,default=16000);p.add_argument('--history-items',type=int,default=20);p.add_argument('--query-retries',type=int,default=3,help='JSON query generation attempts before the plain-text fallback')
    p.add_argument('--campaign-id',default=None);p.add_argument('--resume',action='store_true',help='restore archive+rng from LATEST checkpoint of this campaign id and continue from the next test');p.add_argument('--checkpoint-root',type=Path,default=ROOT/'results'/'checkpoints');p.add_argument('--output',type=Path,default=ROOT/'results'/'C1_v2_direct_llm.json')
    a=p.parse_args();run(a);print(a.output)
if __name__=='__main__':main()
