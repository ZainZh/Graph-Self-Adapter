#!/usr/bin/env python3
"""Compact post-hoc P52 diagnosis summary; arrays remain in source reports."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    teacher=json.loads((a.root/'TEACHER_ACTION_ERRORS.json').read_text())
    rollouts=json.loads((a.root/'ROLLOUT_MODEL_DIAGNOSTIC.json').read_text())
    training=json.loads((a.root/'TRAINING_LOGS.json').read_text())
    grad=json.loads((a.root/'FINAL_CHECKPOINT_GRADIENT_PROBE.json').read_text())
    mse={split:{model:{part:metric['mean'] for part,metric in parts.items()} for model,parts in models.items()} for split,models in teacher['aggregate'].items()}
    cells=[]
    for c in rollouts:
        cells.append(dict(cell=c['cell'],success=c['result']['local_success'],actions=c['result']['actions'],z_drift=c['z_drift'],
            same_state_bc_vs_joint_action=c['same_state_bc_vs_joint_fixed'],online_action_delta=c['same_state_joint_z_effect'],
            online_translation_delta_m=c['same_state_joint_z_translation_m'],online_rotation_delta_rad=c['same_state_joint_z_rotation_rad'],
            window_loss=c['inner'],action_saturation_fraction=c['action_saturation_fraction'],proprio_zscore_rms=c['proprio_zscore_rms']))
    milestones={str(step):{name:next((r for r in value['rows'] if r['update']==step),None) for name,value in training.items()} for step in (100,1000,1100,2000,3000)}
    out=dict(teacher_frame_weighted_mse=mse,matched_training_log_updates=milestones,
        train_joint_to_bc_mse_ratio=mse['train']['joint_fixed']['all']/mse['train']['bc']['all'],
        validation_joint_to_bc_mse_ratio=mse['validation']['joint_fixed']['all']/mse['validation']['bc']['all'],
        final_ten_training_logs={name:{k:v for k,v in value.items() if k!='rows'} for name,value in training.items()},
        final_checkpoint_gradient_probe=grad,cells=cells,
        interpretation_limits=['All data opened development evidence, not fresh heldout',
          'Final-checkpoint first4train gradient probe is not training-history causal evidence',
          'Cosine near zero is near-orthogonal, not negative gradient conflict',
          'Same-observation BC outputs on joint trajectories are counterfactuals, not ground-truth action labels',
          'Loss descent is not useful adaptation; compare induced action changes and closed-loop outcomes'])
    a.output.write_text(json.dumps(out,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(output=str(a.output),teacher_mse=mse)))


if __name__=='__main__':main()
