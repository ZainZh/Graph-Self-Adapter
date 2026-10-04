"""Result-free scoped teacher checks in training domains, no visual policy."""
import json
import numpy as np
import mujoco
from prove.fast_adapter.geometry_insert import GeometryInsert,geometry,domain,initial_position
from prove.fast_adapter.insertion_teacher import reference_action
from tools.fast_adapter.p26_common import R
from tools.fast_adapter.p18_run import save_json


def main():
    out=R/'qualification';out.mkdir(exist_ok=False)
    spec=json.loads((R/'PROTOCOL.json').read_text())['qualification'];rows=[];fresh=[]
    for i in spec['training_ids']:
        e=GeometryInsert(geometry(2201000+i),render=False);e.set_domain(*domain(2202000+i))
        for offset in ([0,0],[.0033,0],[-.0033,.0033]):
            pos=np.r_[e.xy+np.array(offset)*e.scale,.039*e.scale]
            for mode in ('old','repaired'):
                e.reset(pos);assert e.facts()['inserted'];states=[e.state().tolist()];actions=[];flags=[]
                for _ in range(8):
                    action=e.expert_action() if mode=='old' else reference_action(e)
                    e.step(action);mujoco.mj_forward(e.model,e.data)
                    states.append(e.state().tolist());actions.append(action.tolist());flags.append(e.facts()['inserted'])
                rows.append({'domain_id':i,'offset':offset,'teacher':mode,'all_retained':all(flags),'states':states,'actions':actions,'inserted':flags})
        e.reset(initial_position(26300000+i));states=[e.state().tolist()];actions=[];flags=[]
        for _ in range(48):
            action=reference_action(e);e.step(action);mujoco.mj_forward(e.model,e.data)
            states.append(e.state().tolist());actions.append(action.tolist());flags.append(e.facts()['inserted'])
        fresh.append({'domain_id':i,'success':all(flags[-3:]),'states':states,'actions':actions,'inserted':flags});e.close()
    passed=all(r['all_retained'] for r in rows if r['teacher']=='repaired') and all(r['success'] for r in fresh)
    save_json(out/'RESULT.json',{'passed':passed,'old_retained':sum(r['all_retained'] for r in rows if r['teacher']=='old'),
        'repaired_retained':sum(r['all_retained'] for r in rows if r['teacher']=='repaired'),'hold_cases_each':48,
        'repaired_fresh_success':sum(r['success'] for r in fresh),'fresh_cases':16,'physical_actions':1536,
        'scope':'Training-domain qualification only, privileged reference. Original success predicate, no threshold changes.', 'hold_records':rows,'fresh_records':fresh})
    print(json.dumps({'passed':passed,'repaired_hold':sum(r['all_retained'] for r in rows if r['teacher']=='repaired'),'fresh':sum(r['success'] for r in fresh)}),flush=True)
    assert passed,'Teacher qualification failed; do not start training.'

if __name__=='__main__':main()
