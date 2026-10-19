#!/usr/bin/env python3
from __future__ import annotations
import base64, copy, hashlib, json, pickle
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from adaptive_search import CampaignArchive, ExecutedTest, Gene, TransitionStats


def _gene_from_dict(d: Dict[str, Any]) -> Gene:
    params=d.get('params',{}) or {}
    return Gene(str(d.get('rule_id','')),str(d.get('event','')),tuple(sorted((str(k),str(v)) for k,v in params.items())))


def _gene_to_dict(g: Gene) -> Dict[str, Any]:
    return g.to_dict()


def sha256_json(obj: Any) -> str:
    blob=json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()


def encode_random_state(state: Any) -> str:
    return base64.b64encode(pickle.dumps(state,protocol=4)).decode('ascii')


def decode_random_state(text: str) -> Any:
    return pickle.loads(base64.b64decode(text.encode('ascii')))


def serialize_archive(archive: CampaignArchive) -> Dict[str, Any]:
    return {
        'covered_features':sorted(archive.covered_features),
        'transition_stats':{
            k:{'opportunities':v.opportunities,'successes':v.successes}
            for k,v in sorted(archive.transition_stats.items())
        },
        'executed_tests':[
            {
                'target_path':[_gene_to_dict(g) for g in t.target_path],
                'natural_language_test':t.natural_language_test,
                'actual_path':[_gene_to_dict(g) for g in t.actual_path],
                'outcome':t.outcome,
                'execution_log':copy.deepcopy(t.execution_log),
            } for t in archive.executed_tests
        ],
    }


def deserialize_archive(data: Dict[str, Any]) -> CampaignArchive:
    a=CampaignArchive()
    a.covered_features=set(data.get('covered_features',[]))
    a.transition_stats={
        k:TransitionStats(int(v.get('opportunities',0)),int(v.get('successes',0)))
        for k,v in data.get('transition_stats',{}).items()
    }
    for t in data.get('executed_tests',[]):
        a.executed_tests.append(ExecutedTest(
            target_path=[_gene_from_dict(x) for x in t.get('target_path',[])],
            natural_language_test=str(t.get('natural_language_test','')),
            actual_path=[_gene_from_dict(x) for x in t.get('actual_path',[])],
            outcome=str(t.get('outcome','')),
            execution_log=list(t.get('execution_log',[])),
        ))
    return a


class CheckpointRecorder:
    def __init__(self, root: Path, campaign_id: str, model: Dict[str, Any], metadata: Optional[Dict[str,Any]]=None):
        self.root=Path(root)/campaign_id
        self.root.mkdir(parents=True,exist_ok=True)
        self.campaign_id=campaign_id
        self.model=model
        self.base={
            'checkpoint_schema':'1.0',
            'campaign_id':campaign_id,
            'model_id':model.get('model_id'),
            'model_hash':sha256_json(model),
            'metadata':metadata or {},
        }

    def save(self, test_index: int, archive: CampaignArchive, *, last_round: Optional[Dict[str,Any]]=None,
             search_parameters: Optional[Dict[str,Any]]=None, random_state: Any=None) -> Path:
        ck=copy.deepcopy(self.base)
        ck.update({
            'test_index':int(test_index),
            'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'archive':serialize_archive(archive),
            'search_parameters':search_parameters or {},
            'last_round':last_round or {},
        })
        if random_state is not None:
            ck['random_state_b64']=encode_random_state(random_state)
        path=self.root/f'C{test_index:03d}.json'
        path.write_text(json.dumps(ck,ensure_ascii=False,indent=2),encoding='utf-8')
        (self.root/'LATEST').write_text(path.name,encoding='utf-8')
        return path


def load_checkpoint(path: Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def restore_archive(path_or_data: Any) -> CampaignArchive:
    data=load_checkpoint(Path(path_or_data)) if isinstance(path_or_data,(str,Path)) else path_or_data
    return deserialize_archive(data['archive'])
