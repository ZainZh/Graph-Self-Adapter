#!/usr/bin/env python3
"""V34 execution with unambiguous public completion/failure labels."""
from pathlib import Path
import sys,json
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
import three_task_v34 as base
from task_stream_view_v35 import install
install()
import three_task_v26_operator_scope as scopes
from prove.xtrainer.agent import write,digest
original=scopes.new_operator_scope

def scope(*a,**kw):
    result=original(*a,**kw);p=Path(result['scope']);s=json.loads(p.read_text())
    s['display_outcome']='Read public sealed predictions only, distinguish graph complete/failed/unknown; never read private evaluation into execution'
    s['operator_source_bindings'] += [{'executed_path':str(f),'sha256':digest(f)} for f in (Path(__file__).resolve(),HERE/'task_stream_view_v35.py')]
    write(p,s);result['scope_sha256']=digest(p);return result
scopes.new_operator_scope=scope
if __name__=='__main__':base.base.base.main()
