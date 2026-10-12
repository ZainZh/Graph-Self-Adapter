"""New goal-preserving context over frozen compact protocol; no cup goal override."""
import json
from compact_stage_judge import compact_context as original_context,OpenRouterJudge,MODEL
from prove.xtrainer.io import SafetyError

RELATIONS={'inside':'released_inside_declared_target','supported_by_box':'released_supported_by_declared_target','supported_on':'released_supported_by_declared_target','on':'released_supported_by_declared_target','supported':'released_supported_by_declared_target','nested':'released_nested_in_declared_target','hanging':'released_hanging_on_declared_target','drawer_extended':'declared_drawer_extended','toolsheld':'declared_tools_held','handle_supported_by_peg':'released_handle_supported_by_declared_peg','nested_with_support':'released_nested_with_support_in_declared_target','block_inside_dustpan_with_tools_held':'block_inside_declared_dustpan_while_declared_tools_remain_held','receptacle_ready_and_held':'declared_receptacle_ready_and_held','accessible_to_right_arm':'released_stable_in_opposite_arm_accessible_table_region'}


def compact_context(path):
    compact=original_context(path);stage=json.loads(path.read_text())['stage_contract']
    if stage['phase']=='Complete':
        relation=stage['goal_relation']
        if relation not in RELATIONS:raise SafetyError('explicit semantic goal contract missing: '+relation)
        compact.context['goal_relation']=relation;compact.context['query_goal_relation']=RELATIONS[relation]
        compact.context['task_goal_contract']='Confirm released and THIS declared relation, not merely a stationary object near the target. Ambiguity is UNKNOWN.'
    return compact
