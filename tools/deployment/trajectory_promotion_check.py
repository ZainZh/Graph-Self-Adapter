"""Relative V18 receipt verification; no simulator, socket or device imports."""
import hashlib
import json
from pathlib import Path

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check(package_root,receipt_sha256,*,external_paths=None):
    root=Path(package_root).resolve();receipt_path=root/'PROMOTION_RECEIPT.json'
    if digest(receipt_path)!=receipt_sha256:raise ValueError('independent Root-approved receipt SHA mismatch')
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('schema')!='graph-self-adapter.trajectory-promotion-receipt.v1' or receipt.get('status')!='Root-attested-engineering-simulation-pass':raise ValueError('V18 promotion absent')
    names=[row['relative_path'] for row in receipt['files']]
    if len(set(names))!=len(names):raise ValueError('duplicate package bindings')
    for row in receipt['files']:
        path=(root/row['relative_path']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path)!=row['sha256']:raise ValueError('package source/data missing or changed: '+row['relative_path'])
    base=hashlib.sha256(''.join(digest(root/p) for p in receipt['base_runtime_relative_paths']).encode()).hexdigest()
    computed=hashlib.sha256((base+''.join(digest(root/p) for p in receipt['extension_runtime_relative_paths'])).encode()).hexdigest()
    if computed!=receipt['shared_runtime_sha256']:raise ValueError('same V18 runtime identity mismatch')
    checked=[]
    for key,path in (external_paths or {}).items():
        if digest(path)!=receipt['external_resources'][key]['sha256']:raise ValueError('external resource identity mismatch: '+key)
        checked.append(key)
    return {'task':receipt['task'],'shared_runtime_sha256':computed,'package_hashes_verified':True,'external_resources_checked':checked,'simulation_reexecuted':False,'scene_assets_required_for_real_check':False,'device_connections':0,'hardware_commands':0,'physical_qualification_inferred':False,'manual_current_Root_stage_judgement_required':True}
