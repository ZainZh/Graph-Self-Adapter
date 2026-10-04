"""Secondary native grasp evidence and terminal stability; no threshold retuning."""
import json
from pathlib import Path
import numpy as np
from tools.fast_adapter.p28_common import R, TASKS, SEEDS, task_root, save_json, digest
from tools.fast_adapter.p28_eval import EVAL_STRATA


def main():
    records = []
    done = json.loads((R/'CAMPAIGN_DONE.json').read_text())
    for task in done['qualified_tasks']:
        for st in EVAL_STRATA:
            result = json.loads((task_root(task)/'evaluation'/st/'RESULT.json').read_text())
            for row in result['rows']:
                path = Path(row['file']).with_suffix('.json')
                side = json.loads(path.read_text())
                for j, series in enumerate(side['metrics']):
                    first = next((i for i, v in enumerate(series) if v['success']), None)
                    grasp_at = [i for i, v in enumerate(series) if v.get('grasp_success', 0) > .5]
                    records.append({'task': task, 'stratum': st, 'arm': row['arm'],
                        'condition': row['offset']+j, 'success_any': first is not None,
                        'terminal_success': bool(series[-1]['success']),
                        'native_grasp_success_ever': bool(grasp_at) if task == TASKS[0] else None,
                        'grasp_before_or_at_first_task_success': bool(first is not None and any(i <= first for i in grasp_at)) if task == TASKS[0] else None,
                        'source': str(path)})
    summaries = {}
    for task in done['qualified_tasks']:
        summaries[task] = {}
        for st in EVAL_STRATA:
            summaries[task][st] = {}
            for name in ('base', 'fast_correct', 'fast_mean', 'fast_donor', 'history'):
                items = [v for v in records if v['task'] == task and v['stratum'] == st and
                         (v['arm'] == name if name == 'base' else v['arm'].startswith(name+'_seed'))]
                keys = ['success_any', 'terminal_success']
                if task == TASKS[0]:
                    keys += ['native_grasp_success_ever', 'grasp_before_or_at_first_task_success']
                summaries[task][st][name] = {k: 100*float(np.mean([x[k] for x in items])) for k in keys}
    save_json(R/'summary/CONTACT_DIAGNOSTIC.json', {'secondary_descriptive_only': True,
              'no_change_to_preregistered_success_criterion': True,
              'pick_grasp_definition': 'Unmodified native grasp_success: object contact, open-aperture scalar>0, object lifted more than0.02m above initial height.',
              'drawer': 'Native drawer success is reported; grasp_success is not interpreted as grasp evidence for this non-grasping hook/pull task.',
              'summaries_percent': summaries, 'records': records})
    print('CONTACT_DIAGNOSTIC_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
