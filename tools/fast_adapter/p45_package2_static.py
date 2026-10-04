"""Package-2 assembly/visible-pixel check; no task action or success evaluation."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from prove.fast_adapter.robot_geometry_env_p45_v02 import RobotGeometrySkill, configuration, source_assets
from prove.fast_adapter.robot_geometry_env_p45 import RobotGeometrySkill as PackageOne


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def contact_audit(env):
    old = PackageOne(env.cfg, render=False)
    fields = ['geom_contype', 'geom_conaffinity', 'geom_size', 'geom_friction', 'geom_solref', 'geom_solimp', 'geom_margin', 'geom_gap']
    mismatches = []
    declared_changes = []
    physical_count = 0
    try:
        for i in range(old.model.ngeom):
            name = old.model.geom(i).name
            if not name or name == 'robot_pedestal':
                continue
            j = env.model.geom(name).id
            if old.model.geom_contype[i] or old.model.geom_conaffinity[i]:
                physical_count += 1
                for field in fields:
                    if not np.array_equal(getattr(old.model, field)[i], getattr(env.model, field)[j]):
                        if env.task == 'cup' and name == 'table' and field == 'geom_size' and np.array_equal(env.model.geom_size[j], [.35,.36,.025]):
                            declared_changes.append(dict(name=name,field=field,old=old.model.geom_size[i].tolist(),new=env.model.geom_size[j].tolist()))
                        else:
                            mismatches.append(f'{name}.{field}')
        for i in range(old.model.nbody):
            name = old.model.body(i).name
            if not name:
                continue
            j = env.model.body(name).id
            for field in ['body_mass', 'body_inertia']:
                if not np.array_equal(getattr(old.model, field)[i], getattr(env.model, field)[j]):
                    mismatches.append(f'{name}.{field}')
        visual = [i for i in range(env.model.ngeom) if env.model.geom(i).name.startswith(('frame_front_', 'frame_surface_'))]
        return dict(shared_physical_geometry_checked=physical_count, physics_mismatches=mismatches,
                    declared_scene_changes=declared_changes,
                    new_visual_surface_count=len(visual), all_visual_surfaces_noncolliding=all(env.model.geom_contype[i] == 0 and env.model.geom_conaffinity[i] == 0 for i in visual),
                    opposing_finger_exclusion_count=int(env.model.nexclude), no_undeclared_physics_changes=not mismatches,
                    disclosed_installation_change='cup-family base z -.18 to -.35, matching pedestal and table x half-width .44 to .35; canonical targets unchanged')
    finally:
        old.close()


def main(out, tasks):
    out = Path(out); out.mkdir(parents=True, exist_ok=False)
    sources = ['tools/fast_adapter/p45_package2_static.py', 'prove/fast_adapter/robot_geometry_env_p45_v02.py', 'prove/fast_adapter/robot_geometry_env_p45.py', 'prove/fast_adapter/p45_handle_surfaces.py', 'prove/fast_adapter/p45_visible_frame.py', 'prove/fast_adapter/geometry3d_env.py']
    pre = dict(scope='package2 static assembly and declared reset view; constructor reset settling only, no task actions; not engineering qualification', tasks=tasks, sources={p:sha(p) for p in sources}, assets=source_assets())
    (out/'PRE_RUN.json').write_text(json.dumps(pre,indent=2)+'\n')
    rows = []
    for task in tasks:
        base = 450000 if task == 'cup' else 451000
        for split, offset in [('id',0),('geometry',100)]:
            for i in range(4):
                seed=base+offset+i; folder=out/f'{task}_{split}_{seed}';folder.mkdir()
                e=RobotGeometrySkill(configuration(task,seed,split),render=True)
                try:
                    obs,rgb,mask=e.observe()
                    Image.fromarray(rgb).save(folder/'wrist.png');Image.fromarray(mask*255).save(folder/'visible_goal.png');Image.fromarray(e.render_global()).save(folder/'global.png')
                    np.save(folder/'OBS.npy',obs,allow_pickle=True)
                    error=None
                    if obs['corners'] is not None:
                        p,R,half=e.oracle()
                        if task=='insert': half=half+.006;p=p+R[:,2]*.0006
                        corners=np.array([[-half[0],half[1],0],[half[0],half[1],0],[half[0],-half[1],0],[-half[0],-half[1],0]])@R.T+p
                        cam=(corners-obs['camera_position'])@obs['camera_rotation'];K=obs['K'];uv=np.c_[K[0,2]+K[0,0]*cam[:,0]/(-cam[:,2]),K[1,2]-K[1,1]*cam[:,1]/(-cam[:,2])]
                        error=min(float(np.sqrt(np.square(np.roll(obs['corners'],k,axis=0)-uv).sum(-1).mean())) for k in range(4))
                    row=dict(task=task,seed=seed,split=split,visible=obs['corners'] is not None,corner_rms_px=error,goal_pixels=int(mask.sum()),robot=e.robot_tracking_diagnostics())
                    if i==0 and split=='id': row['contact_audit']=contact_audit(e)
                    if task=='cup':
                        alpha=e.model.geom_rgba[:,3].copy()
                        for j in range(e.model.ngeom):
                            name=e.model.geom(j).name
                            if name.startswith(('frame_front_','frame_surface_')):e.model.geom_rgba[j,3]=0
                            elif name.startswith('handle_'):e.model.geom_rgba[j,3]=1
                        e.renderer.disable_segmentation_rendering();e.renderer.update_scene(e.data,camera='wrist');before=e.renderer.render().copy()
                        e.model.geom_rgba[:,3]=alpha
                        Image.fromarray(before).save(folder/'wrist_original_unsplit_surface.png')
                        diff=np.abs(rgb.astype(float)-before.astype(float));row['surface_partition_rgb_mean_abs_255']=float(diff.mean());row['surface_partition_rgb_changed_fraction']=float((diff.max(-1)>1).mean())
                    rows.append(row);print(json.dumps(row),flush=True)
                finally:e.close()
    (out/'RESULTS.json').write_text(json.dumps(rows,indent=2)+'\n')
    (out/'SEAL.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='SEAL.json'},indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--tasks',nargs='+',choices=['cup','insert'],default=['cup','insert']);a=p.parse_args();main(a.out,a.tasks)
