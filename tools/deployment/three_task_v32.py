#!/usr/bin/env python3
"""Common V27 live interface with automatic Astra API initial grounding."""
import sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import three_task_v26 as v26
v26.DEFAULT_REGISTRY=HERE.parents[1]/'evaluations/api_grounding_v32/REGISTRY.json'
import three_task_v26_operator_scope as scopes
from prove.xtrainer.agent import write,digest
_original_scope=scopes.new_operator_scope
def api_scope(*args,**kwargs):
    result=_original_scope(*args,**kwargs);path=Path(result['scope']);scope=__import__('json').loads(path.read_text())
    scope.update(model='openai/gpt-6-astra',endpoint='https://openrouter.ai/api/v1/chat/completions',judge_fallback='No manual Codex fallback; API UNKNOWN stops under original freshness guard',grounding_provider='openrouter/openai/gpt-6-astra',stage_judge_provider='openrouter/openai/gpt-6-astra',human_root_required=False,API_grounding_maximum_calls=2,retained_pre_lift_public_frame=True,revision='v30 bounded unconfirmed closure-Lift observation; original Lift displacement; only after current API holding may Move execute',qualification='New engineering API candidate, no borrowed parent full-task qualification')
    scope['operator_source_bindings'] += [{'executed_path':str(p),'sha256':digest(p)} for p in (Path(__file__).resolve(),HERE/'api_grounding_v32.py',HERE/'api_stage_judge_v32.py',HERE/'api_v32_apple_entry.py',HERE/'apple_probe_lift_v30_runtime.py')]
    write(path,scope);result['scope_sha256']=digest(path);return result
scopes.new_operator_scope=api_scope
from three_task_v27 import main
if __name__=='__main__':main()
