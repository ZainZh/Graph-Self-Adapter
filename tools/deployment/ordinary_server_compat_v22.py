#!/usr/bin/env python3
"""Additive compatible-SAM/evaluator candidate; all old server freezes preserved."""
import json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
for path in (HERE,REPO):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
import ordinary_server_fair_v22 as fair
from prove.xtrainer.agent import digest
from prove.xtrainer.io import SafetyError
base=fair.base;THREAD_ENV=fair.THREAD_ENV
MANIFEST=REPO/'evaluations/multitask_parallel_v22/dual_tool/server_shared/REMOTE_REPAIR_MANIFEST.json'
EXPECTED='95293256010ce6f5f67b5afea39bc7b25ea02647c0c45536dccb19030f04393b'
_installed=False


def install():
    global _installed
    if _installed:return
    if digest(MANIFEST)!=EXPECTED:raise SafetyError('shared compatible backend manifest changed')
    repair=json.loads(MANIFEST.read_text());targets=[]
    for row in repair['records']:
        path=Path(row['target'])
        if digest(path)!=row['target_sha256']:raise SafetyError('active repaired dependency changed')
        targets.append(path)
    parity=REPO/'evaluations/multitask_parallel_v22/server_shared/ROOT_SAM_COMPONENT_PARITY.json'
    proof=json.loads(parity.read_text())
    if proof.get('status')!='passed' or proof.get('candidate_sha256')!=digest(REPO.parent/'SAM2_streaming/sam2/_C.py'):raise SafetyError('active SAM component parity receipt missing/mismatch')
    completion=REPO/'evaluations/multitask_parallel_v22/server_shared/ASSET_COMPLETION_MANIFEST.json'
    if digest(completion)!='769db9a800528d28dbff9b1ea0286b400448ce1d2721ee0918021bc33be0add1':raise SafetyError('completed XML/evaluator/SAM readiness manifest mismatch')
    fair.install();base.SOURCES=base.SOURCES+(Path(__file__),MANIFEST,parity,completion,*targets)
    _installed=True


def main(argv=None):
    install();return fair.main(argv)

if __name__=='__main__':main()
