#!/usr/bin/env python3
"""Laptop file-only promotion identity check; no scene, device or connection."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check(package_root,receipt_sha256,*,external_paths=None):
    root=Path(package_root).resolve();receipt_path=root/'PROMOTION_RECEIPT.json'
    if digest(receipt_path)!=receipt_sha256:raise ValueError('independent Root-approved receipt SHA mismatch')
    receipt=json.loads(receipt_path.read_text())
    if receipt.get('schema')!='graph-self-adapter.macro-promotion-receipt.v1' or receipt.get('status')!='Root-attested-engineering-simulation-pass':raise ValueError('Root engineering promotion receipt absent')
    names=[row['relative_path'] for row in receipt['files']]
    if len(set(names))!=len(names) or not set(receipt['runtime_relative_paths'])<=set(names):raise ValueError('duplicate or undeclared runtime source')
    for row in receipt['files']:
        path=(root/row['relative_path']).resolve()
        if not path.is_relative_to(root) or not path.is_file() or digest(path)!=row['sha256']:raise ValueError('package source/data missing or changed: '+row['relative_path'])
    computed=hashlib.sha256(''.join(digest(root/path) for path in receipt['runtime_relative_paths']).encode()).hexdigest()
    if computed!=receipt['shared_runtime_sha256']:raise ValueError('same shared runtime identity mismatch')
    checked=[]
    for key,path in (external_paths or {}).items():
        if key not in receipt['external_resources'] or digest(path)!=receipt['external_resources'][key]['sha256']:raise ValueError('external resource identity mismatch: '+key)
        checked.append(key)
    return {'task':receipt['task'],'shared_runtime_sha256':computed,'package_hashes_verified':True,'external_resources_checked':checked,
        'simulation_reexecuted':False,'simulation_evidence_basis':'Root workstation verified original frozen complete success; receipt SHA independently supplied',
        'scene_assets_required_for_real_check':False,'device_connections':0,'hardware_commands':0,'physical_qualification_inferred':False,
        'remaining_activation':'separate reviewed real scope/config and explicit operator command declaration; current Root manual RGB boundary replies'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--receipt-sha256',required=True)
    p.add_argument('--sam-checkpoint',type=Path);p.add_argument('--sam-utility',type=Path);a=p.parse_args(argv)
    external={k:v for k,v in {'sam_checkpoint':a.sam_checkpoint,'sam_utility':a.sam_utility}.items() if v is not None}
    print(json.dumps(check(a.package,a.receipt_sha256,external_paths=external),indent=2))

if __name__=='__main__':main()
