#!/usr/bin/env python3
from __future__ import annotations
import json, os, shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX_PATH = ROOT / 'nlp_v1_1' / 'libraries' / 'gaia_file_index.json'
DEFAULT_CATALOG_PATH = ROOT / 'nlp_v1_1' / 'libraries' / 'entity_fact_catalog.json'
DEFAULT_PROVISION_DIR = ROOT / 'results' / 'provisioned_files'


def resolve_gaia_root(explicit: Optional[Path] = None) -> Optional[Path]:
    candidates: List[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env = os.getenv('MAS_GAIA_2023')
    if env:
        candidates.append(Path(env))
    candidates += [ROOT / 'data' / 'GAIA' / '2023',
                   ROOT.parent / 'code-0919' / 'data' / 'GAIA' / '2023']
    for c in candidates:
        if c.is_dir():
            return c
    return None


class FileIndex:

    def __init__(self, index_path: Path = DEFAULT_INDEX_PATH):
        self.index_path = Path(index_path)
        data = json.loads(self.index_path.read_text(encoding='utf-8')) if self.index_path.exists() else {'entries': {}}
        self.entries: Dict[str, List[Dict[str, str]]] = data.get('entries', {})

    def lookup(self, question_id: Optional[str]) -> List[Dict[str, str]]:
        if not question_id:
            return []
        return list(self.entries.get(str(question_id), []))


def _question_id_of(record: Dict[str, Any]) -> Optional[str]:
    return ((record.get('oracle') or {}).get('evidence') or {}).get('question_id')


def seed_file_map(
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    gaia_root: Optional[Path] = None,
    index: Optional[FileIndex] = None,
) -> Dict[str, Dict[str, str]]:
    root = resolve_gaia_root(gaia_root)
    if root is None:
        return {}
    index = index or FileIndex()
    catalog = json.loads(Path(catalog_path).read_text(encoding='utf-8'))
    out: Dict[str, Dict[str, str]] = {}
    for r in catalog.get('records', []):
        if not r.get('benchmark_record'):
            continue
        qid = _question_id_of(r)
        if not qid or qid in out:
            continue
        for hit in index.lookup(qid):
            host = root / hit['split'] / hit['file_name']
            if host.exists():
                out[qid] = {'file_name': hit['file_name'], 'split': hit['split'], 'host_path': str(host)}
                break
    return out


def provision_files(
    provision_dir: Path = DEFAULT_PROVISION_DIR,
    *,
    catalog_path: Path = DEFAULT_CATALOG_PATH,
    gaia_root: Optional[Path] = None,
    sandbox_root: Optional[str] = None,
    index: Optional[FileIndex] = None,
) -> Dict[str, Dict[str, str]]:
    provision_dir = Path(provision_dir)
    provision_dir.mkdir(parents=True, exist_ok=True)
    m = seed_file_map(catalog_path, gaia_root, index)
    visible = sandbox_root or str(provision_dir.resolve())
    for info in m.values():
        dst = provision_dir / info['file_name']
        if not dst.exists():
            shutil.copy2(info['host_path'], dst)
        info['sandbox_path'] = f"{visible.rstrip('/')}/{info['file_name']}"
    return m


def sandbox_hint(info: Dict[str, str]) -> str:
    return f"The referenced file is available inside the sandbox at: {info['sandbox_path']}"


def dependency_for_fact(grounded_fact: Dict[str, Any], seed_map: Dict[str, Dict[str, str]]) -> Optional[Dict[str, str]]:
    qid = _question_id_of(grounded_fact or {})
    info = seed_map.get(qid) if qid else None
    if not info:
        return None
    return {'question_id': qid, 'file_name': info['file_name'], 'split': info.get('split', ''),
            'host_path': info['host_path'], 'sandbox_path': info['sandbox_path'],
            'hint': sandbox_hint(info)}


def apply_hint(query: str, dep: Optional[Dict[str, str]]) -> str:
    if not dep:
        return query
    return f"{query}\n\n{dep['hint']}"
