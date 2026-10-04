"""Read-only P64 cup diagnosis: replay public gates, never robot physics.

Evaluator-only geometry is reconstructed after sealing from known scene
construction and recorded cup free-joint pose. It never enters runtime code.
Same-frame z=0 is a numerical counterfactual, not a physical outcome claim.
"""
import json
from pathlib import Path
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from prove.fast_adapter.p64_cup import configuration,NOMINAL_ROTATION
from prove.fast_adapter.robot_geometry_env_p45_v02 import GeometryTaskController
from prove.fast_adapter.p61_data import sha256
from tools.fast_adapter.p64_cup_report import load_cell
from prove.fast_adapter.p64_cup_policy import CupActionPolicy,PHASES


def stats(values):
    a=np.asarray(values,float)
    return dict(n=len(a),mean=float(a.mean()),median=float(np.median(a)),p95=float(np.quantile(a,.95)),max=float(a.max())) if len(a) else None


def run(report,output):
    report,output=Path(report),Path(output)
    if output.exists():raise FileExistsError(output)
    summary=json.loads(report.read_text());rows=[]
    action_path=report.parent.parent/'action_v01/final.pt'
    action=CupActionPolicy();action.load_state_dict(torch.load(action_path,map_location='cpu',weights_only=True)['state_dict'])
    action.eval().requires_grad_(False)
    for source in summary['cells']:
        cell=load_cell(source['path']);path=Path(cell['path']);result=cell['result'];condition=result['cell']
        observations=np.load(path/'OBS.npy',allow_pickle=True).tolist();trace=cell['trace']
        neural_rows=[r for r in trace if r['neural_action']['used']]
        with torch.no_grad():
            inputs=[torch.tensor(np.stack([r['neural_action'][key] for r in neural_rows]),dtype=torch.float32) for key in ('center','size')]
            inputs.append(torch.tensor(np.stack([observations[0]['position']]*len(neural_rows)),dtype=torch.float32))
            inputs.append(torch.tensor(np.stack([r['neural_action']['anchor'] for r in neural_rows]),dtype=torch.float32))
            inputs.append(torch.tensor([PHASES.index(r['neural_action']['phase']) for r in neural_rows]))
            replayed=action(*inputs).numpy()
        actual_commands=np.stack([r['command_p'] for r in neural_rows])
        errors=np.linalg.norm(replayed-actual_commands,axis=-1)
        command_audit=dict(n=len(neural_rows),max_error_m=float(errors.max()),mean_error_m=float(errors.mean()),
            tolerance_m=1e-6,passed=bool(errors.max()<=1e-6),method='independent CPU batched checkpoint forward versus saved actual command_p')
        if not command_audit['passed']:raise ValueError('Actual MLP command recomputation failed')
        controller=GeometryTaskController('cup',observations[0]['position'],observations[0]['rotation'])
        cfg=configuration('cup',condition['seed'],condition['split'])
        # Scene construction inserts the object/freejoint before Panda joints.
        # Verify record dimensionality, initial XY and normalized quaternion.
        q0=np.asarray(trace[0]['state']['qpos'])
        if len(q0)!=16 or not np.allclose(q0[:2],cfg['center'][:2],atol=1e-5) or abs(np.linalg.norm(q0[3:7])-1)>1e-5:
            raise ValueError('Cup sealed freejoint layout not verified')
        gates=[];geometry=[];stage_match=True
        for row,obs in zip(trace,observations):
            prior=controller.phase
            target,R,closed,stage=controller.command(row['t'],row['estimate'],(obs['position'],obs['rotation']))
            stage_match=stage_match and stage==row['stage'] and bool(closed)==row['closed']
            if row['neural_action']['used']:
                phase=row['neural_action']['phase'];threshold=.005 if phase=='close' else (.003 if phase=='approach' else .004)
                angle_limit=.06 if phase=='approach' else .07
                actual=np.asarray(obs['position']);command=np.asarray(row['command_p'])
                angle=np.linalg.norm(Rotation.from_matrix(R@np.asarray(obs['rotation']).T).as_rotvec())
                gates.append(dict(t=row['t'],phase=phase,neural_to_gate_mm=float(np.linalg.norm(command-target)*1000),
                    own_to_gate_mm=float(np.linalg.norm(actual-target)*1000),own_to_neural_mm=float(np.linalg.norm(actual-command)*1000),
                    gate_mm=threshold*1000,own_to_gate_angle_rad=float(angle),angle_limit_rad=angle_limit,
                    position_gate_pass=bool(np.linalg.norm(actual-target)<threshold),angle_gate_pass=bool(angle<angle_limit)))
            record=row.get('adaptation')
            if record is None or record['t']!=row['t']:continue
            base=np.asarray(record['geometry8']);z=np.asarray(record['z_after']);anchor=np.asarray(record['T_world_anchor'])
            center0=anchor[:3,:3]@base[:3]+anchor[:3,3]
            center1=anchor[:3,:3]@(base[:3]+.02*z[:3])+anchor[:3,3]
            size0=np.exp(base[6:]);size1=np.exp(base[6:]+.25*z[6:])
            q=np.asarray(row['state']['qpos']);object_R=Rotation.from_quat(np.r_[q[4:7],q[3]]).as_matrix()
            truth_center=q[:3]+object_R@np.array([cfg['radius']+cfg['handle_width']/2,-.004,cfg['handle_z']])
            truth_size=np.array([cfg['handle_width']/2+.003,cfg['handle_height']/2+.003])
            geometry.append(dict(t=row['t'],center0_mm=float(np.linalg.norm(center0-truth_center)*1000),
                center_online_mm=float(np.linalg.norm(center1-truth_center)*1000),
                outer_halfsize0_mm=float(np.linalg.norm(size0-truth_size)*1000),
                outer_halfsize_online_mm=float(np.linalg.norm(size1-truth_size)*1000),
                corrected_center_world=center1.tolist(),raw_center_world=center0.tolist(),truth_center_world=truth_center.tolist(),
                corrected_halfsize=size1.tolist(),raw_halfsize=size0.tolist(),truth_halfsize=truth_size.tolist(),
                latent_norm=float(np.linalg.norm(z)),U=record['U']))
        if not stage_match:raise ValueError('Public stage replay differs from recorded stage: '+str(condition))
        phase_summary={}
        for phase in sorted({g['phase'] for g in gates}):
            gg=[g for g in gates if g['phase']==phase];tail=gg[-min(30,len(gg)):]
            phase_summary[phase]=dict(steps=len(gg),last_t=gg[-1]['t'],
                tail_neural_to_gate_mm=stats([g['neural_to_gate_mm'] for g in tail]),
                tail_own_to_gate_mm=stats([g['own_to_gate_mm'] for g in tail]),
                tail_own_to_neural_mm=stats([g['own_to_neural_mm'] for g in tail]),
                tail_position_gate_pass=sum(g['position_gate_pass'] for g in tail),
                tail_angle_gate_pass=sum(g['angle_gate_pass'] for g in tail),tail_n=len(tail))
        first,last=geometry[0],geometry[-1]
        rows.append(dict(cell=condition,validated_success=result['validated_success'],outcome_class=result['outcome_class'],
            final_phase=result['final_phase'],stage_replay_exact=True,phase_summary=phase_summary,independent_command_audit=command_audit,
            final_public_gate=gates[-1],first_geometry=first,last_geometry=last,
            same_frame_counterfactual_summary={k:stats([g[k] for g in geometry]) for k in ('center0_mm','center_online_mm','outer_halfsize0_mm','outer_halfsize_online_mm')},
            geometry_drift_mm=float(np.linalg.norm(np.asarray(last['corrected_center_world'])-first['corrected_center_world'])*1000),
            raw_geometry_drift_mm=float(np.linalg.norm(np.asarray(last['raw_center_world'])-first['raw_center_world'])*1000),
            geometric_records=len(geometry),seal_sha256=sha256(path/'SEAL.json')))
    result=dict(schema='prove.p64-cup-postseal-diagnostic.v1',report_sha256=sha256(report),source_sha256=sha256(__file__),
        physics_steps=0,optimizer_updates=0,action_checkpoint_sha256=sha256(action_path),controller_source='unchanged public P45 gate replay using recorded EEF/estimates',
        private_truth='postseal only: frozen condition geometry plus sealed object freejoint pose, verified16Dqpos/XY/quaternion',
        limits=['Numerical same-frame z0 versus adapted geometry is not a robot success counterfactual.',
                'Observed gate mismatch identifies an execution obstruction; it does not isolate every cause of the paired outcome.',
                'Maximum stage drift includes changing camera-dependent encoder output and accumulated latent; it is not latent-only drift.'],
        selected_pairs=[p for p in summary['pairs'] if p['harm'] or p['rescue']],cells=rows)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--report',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();run(a.report,a.output)
