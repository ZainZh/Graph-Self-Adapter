#!/usr/bin/env python3
"""Evaluator-only P52 contact reconstruction at <=12 sealed key states.

Reads entry0/1 × three arms × first/last. mj_fwdPosition only: no env, action,
integration, training, renderer or trajectory generation. Original seals untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from prove.task_graph.libero_supervisor import contact_facts


def main():
    import mujoco
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    rows=[];model=None;xml_sha=None
    for path in sorted(a.root.rglob('RESULT.json')):
        result=json.loads(path.read_text())
        if result.get('cell',{}).get('entry') not in (0,1):continue
        folder=path.parent
        if not (folder/'public/ACTIONS.jsonl').exists():continue
        binding=json.loads((folder/'BINDING.json').read_text());xml=binding['environment']['relocated_xml'];digest=hashlib.sha256(xml.encode()).hexdigest()
        if model is None:model=mujoco.MjModel.from_xml_string(xml);xml_sha=digest
        if digest!=xml_sha:raise ValueError('Representative cells use different XML')
        names={kind:[mujoco.mj_id2name(model,getattr(mujoco.mjtObj,'mjOBJ_'+kind.upper()),i) for i in range(getattr(model,'n'+('body' if kind=='body' else 'geom')))] for kind in ('body','geom')}
        book=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'black_book_1_main')
        caddy=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_SITE,'desk_caddy_1_back_contain_region')
        trace=[json.loads(s) for s in (folder/'public/ACTIONS.jsonl').read_text().splitlines()]
        for t in (trace[0]['t'],trace[-1]['t']):
            state=folder/'private'/f'frame_{t:04d}.npz';data=mujoco.MjData(model)
            with np.load(state,allow_pickle=False) as saved:
                data.qpos[:]=saved['supervisor_qpos']
                # Position-only contact geometry: no force/integration calculation.
                mujoco.mj_fwdPosition(model,data)
                body_error=float(np.max(np.abs(data.xpos-saved['supervisor_body_xpos'])))
                if body_error>1e-7:raise ValueError('Reconstructed body poses disagree with sealed state')
                ctrl=saved['supervisor_ctrl'].copy()
            contacts=[dict(geom1=int(data.contact[i].geom1),geom2=int(data.contact[i].geom2),distance=float(data.contact[i].dist)) for i in range(data.ncon)]
            fingers,wrong=contact_facts(names,contacts)
            relevant=[dict(geom1=names['geom'][c['geom1']],geom2=names['geom'][c['geom2']],distance=c['distance']) for c in contacts if any((names['geom'][c[g]] or '').startswith('black_book_1') for g in ('geom1','geom2'))]
            R=data.site_xmat[caddy].reshape(3,3);B=data.xmat[book].reshape(3,3)
            joint=[]
            for j in range(model.njnt):
                name=mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_JOINT,j) or ''
                if not name.startswith('robot0_joint'):continue
                q=float(data.qpos[model.jnt_qposadr[j]]);lo,hi=map(float,model.jnt_range[j])
                joint.append(dict(name=name,qpos=q,range=[lo,hi],margin_to_limit=min(q-lo,hi-q)))
            limited=np.asarray(model.actuator_ctrllimited,dtype=bool);ranges=np.asarray(model.actuator_ctrlrange)
            saturation=((ctrl<=ranges[:,0]+1e-6)|(ctrl>=ranges[:,1]-1e-6))&limited
            rows.append(dict(cell=result['cell'],t=t,endpoint='first' if t==trace[0]['t'] else 'last',local_success=result['local_success'],
                source_state=str(state),source_sha256=hashlib.sha256(state.read_bytes()).hexdigest(),body_pose_max_error=body_error,
                reconstructed_finger_book_contacts=fingers,reconstructed_wrong_target=wrong,book_contacts=relevant,
                book_in_target_translation=(R.T@(data.xpos[book]-data.site_xpos[caddy])).tolist(),
                book_in_target_rotation_rotvec=Rotation.from_matrix(R.T@B).as_rotvec().tolist(),robot_joint_limits=joint,
                actuators_at_ctrl_limit=[mujoco.mj_id2name(model,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in np.flatnonzero(saturation)]))
    if len(rows)>12:raise ValueError('Exceeded predeclared representative state budget')
    report=dict(scope='Reconstructed geometric contacts at sealed existing states, not directly logged contacts. No integration or action.',
                states=len(rows),xml_sha256=xml_sha,mujoco_version=mujoco.__version__,rows=rows)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(states=len(rows),output=str(a.output))))


if __name__=='__main__':main()
