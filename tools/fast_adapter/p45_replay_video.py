"""Re-render complete sealed P45 neural states with zero physics integration.

This creates explicitly labeled reconstruction derivatives, never original
recordings or a replacement trajectory seal. No model or controller is run.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib
import importlib.metadata
from pathlib import Path
import sys

import numpy as np

from tools.fast_adapter.p45_video import ROOT, read_json, write_json, sha

SCHEMA='prove.p45-state-render.v1'
ORIGIN_FILES=['PRE_RUN.json','SEAL.json','TRACE.npy','TERMINAL.npy','RESULT.json','OBS.npy']
ALLOWED_BACKENDS={
    ('prove.fast_adapter.robot_geometry_env_p45_v02','RobotGeometrySkill'),
    ('prove.fast_adapter.p45_slider_env_v03','SliderGeometrySkill'),
}


def validate_origin(episode):
    episode=Path(episode).resolve()
    pre,seal,result=[read_json(episode/name) for name in ['PRE_RUN.json','SEAL.json','RESULT.json']]
    if seal.get('schema')!='prove.p45-trajectory-seal.v1':
        raise ValueError('Only complete P45 neural trajectory seals are supported')
    for name in ['PRE_RUN.json','TRACE.npy','TERMINAL.npy','OBS.npy']:
        if seal['files'].get(name)!=sha(episode/name):raise ValueError(f'Sealed input changed: {name}')
    if seal.get('neural_parameters_before')!=seal.get('neural_parameters_after'):
        raise ValueError('Neural weights changed in the source execution')
    frozen_path=Path(pre['freeze']['path'])
    if not frozen_path.is_absolute():frozen_path=ROOT/frozen_path
    if sha(frozen_path)!=pre['freeze']['sha256']:raise ValueError('Source freeze changed')
    frozen=read_json(frozen_path);cell=pre['cell']
    if frozen['cells'][pre['cell_index']]!=cell:raise ValueError('Cell identity changed')
    if pre['environment']!=frozen['environments'][cell['task']]:raise ValueError('Environment binding changed')
    backend=pre['environment']
    if (backend['module'],backend['class']) not in ALLOWED_BACKENDS:
        raise ValueError('Reconstruction is limited to the currently frozen P45 v02 / slider v03 backends')
    for key in ['task','split','condition_seed','rep','model_seed','arm']:
        if result.get(key)!=cell[key]:raise ValueError(f'Result identity differs: {key}')
    if any(type(result.get(key)) is not bool for key in ['validated_success','mechanical_valid']):
        raise ValueError('Source episode has no complete evaluated outcome')
    for name,digest in frozen['sources'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts or sha(ROOT/relative)!=digest:
            raise ValueError(f'Current source does not match the source freeze: {name}')
    for name in [backend[key].replace('.','/')+'.py' for key in ['module','configuration_module','controller_module']]:
        if name not in frozen['sources']:raise ValueError(f'Backend source is outside the freeze: {name}')
    return episode,pre,seal,result,frozen,frozen_path


@contextmanager
def integration_guard(mujoco):
    """Count constructor no-ops; reject any integration after construction."""
    names=['mj_step','mj_step1','mj_step2']
    original={name:getattr(mujoco,name) for name in names}
    receipt=dict(constructor_active=True,suppressed_constructor_calls=0,
                 forbidden_integration_attempts=0,physics_steps_executed=0)
    def blocked(*args,**kwargs):
        receipt['forbidden_integration_attempts']+=1
        raise RuntimeError('Physics integration is forbidden in sealed-state reconstruction')
    def constructor_step(*args,**kwargs):
        if not receipt['constructor_active']:return blocked(*args,**kwargs)
        receipt['suppressed_constructor_calls']+=1
    mujoco.mj_step=constructor_step
    mujoco.mj_step1=blocked;mujoco.mj_step2=blocked
    try:yield receipt
    finally:
        for name,function in original.items():setattr(mujoco,name,function)


def restore_integration(mujoco,model,data,state):
    """Restore the full sealed integration state, then refresh render fields."""
    spec=mujoco.mjtState.mjSTATE_INTEGRATION
    integration=np.asarray(state['integration'],dtype=float)
    expected=int(mujoco.mj_stateSize(model,spec))
    if integration.shape!=(expected,) or not np.isfinite(integration).all():
        raise ValueError('Invalid sealed full integration state')
    mujoco.mj_setState(model,data,integration,spec)
    def assert_primary_state():
        if float(data.time)!=float(state['time']):raise ValueError('Sealed integration/time mismatch')
        for key in ['qpos','qvel','act','ctrl']:
            if key in state and not np.array_equal(np.asarray(getattr(data,key)),np.asarray(state[key])):
                raise ValueError(f'Sealed integration/{key} mismatch or forward changed primary state')
    assert_primary_state()
    mujoco.mj_forward(model,data)
    assert_primary_state()
    after=np.empty(expected);mujoco.mj_getState(model,data,after,spec)
    return float(np.max(np.abs(after-integration),initial=0.))


def dependency_inventory():
    versions={}
    for name in ['mujoco','numpy','scipy','Pillow','imageio','imageio-ffmpeg']:
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    modules={}
    # The entire frozen repository inventory is recorded separately. Hash the
    # loaded MuJoCo wrapper/renderer/native files as the rendering runtime.
    for name,module in sorted(sys.modules.items()):
        file=getattr(module,'__file__',None)
        if name.startswith('mujoco') and file and Path(file).is_file():
            modules[name]=dict(path=str(Path(file).resolve()),sha256=sha(file))
    package=Path(sys.modules['mujoco'].__file__).resolve().parent
    native={str(file):sha(file) for file in sorted(package.glob('*.so*')) if file.is_file()}
    return dict(package_versions=versions,loaded_mujoco_modules=modules,mujoco_native_libraries=native)


def reconstruct(episode,out):
    episode,pre,seal,result,frozen,frozen_path=validate_origin(episode)
    out=Path(out).resolve()
    if out==episode or episode in out.parents:raise ValueError('Reconstruction must be outside the source episode')
    out.mkdir(parents=True,exist_ok=False)
    trace=np.load(episode/'TRACE.npy',allow_pickle=True).tolist()
    terminal=np.load(episode/'TERMINAL.npy',allow_pickle=True).item()
    if len(trace)!=pre['steps'] or [row['t'] for row in trace]!=list(range(pre['steps'])):
        raise ValueError('Incomplete source trace')
    states=[row['state'] for row in trace]+[terminal]
    if any('integration' not in state for state in states):raise ValueError('Complete integration snapshots required')
    manifest=dict(schema=SCHEMA,status='started',source_episode=str(episode),
                  origin_files={name:sha(episode/name) for name in ORIGIN_FILES},
                  source_freeze=dict(path=str(frozen_path),sha256=sha(frozen_path)),cell=pre['cell'],
                  renderer_sources={str(Path(__file__).relative_to(ROOT)):sha(__file__),
                                    'tools/fast_adapter/p45_video.py':sha(ROOT/'tools/fast_adapter/p45_video.py')},
                  frozen_source_inventory=frozen['sources'],original_seal_unchanged=True,
                  no_adapter_or_task_controller_execution=True,
                  construction='original frozen constructor; mj_step is a counted no-op only during construction; mj_step1/2 always raise',
                  frame_identity='封存状态重建回放；不是当时保存的原始 RGB',
                  derived_state_caveat='mj_forward recomputes render fields at sealed qpos; original post-mj_step fields can lag by one integration step (normally 1ms). No claim of pixel-identical reconstruction.',
                  physics_steps_executed=0)
    write_json(out/'MANIFEST.json',manifest)
    import mujoco
    backend=pre['environment'];module=importlib.import_module(backend['module'])
    configuration=importlib.import_module(backend['configuration_module']).configuration
    robot_module=importlib.import_module(backend['controller_module'])
    if robot_module.source_assets()!=frozen['robot_assets']:
        raise ValueError('Robot XML/mesh assets differ from source execution')
    manifest['robot_assets']=frozen['robot_assets']
    env=None;frames=[];details=[];integration_deltas=[]
    try:
        with integration_guard(mujoco) as receipt:
            env=getattr(module,backend['class'])(configuration(pre['cell']['task'],pre['cell']['condition_seed'],pre['cell']['split']),render=True)
            receipt['constructor_active']=False
            # Any accidental normal task/step call is separately blocked too.
            def forbidden_task(*args,**kwargs):raise RuntimeError('New task execution is forbidden in reconstruction')
            env.step=forbidden_task
            for state in states:
                integration_deltas.append(restore_integration(mujoco,env.model,env.data,state))
                env.renderer.disable_segmentation_rendering()
                env.renderer.update_scene(env.data,camera='global');global_rgb=env.renderer.render().copy()
                env.renderer.update_scene(env.data,camera='wrist');wrist_rgb=env.renderer.render().copy()
                env.renderer.update_scene(env.data,camera='detail');details.append(env.renderer.render().copy())
                frames.append(np.concatenate([global_rgb,wrist_rgb],axis=1))
            manifest.update({key:value for key,value in receipt.items() if key!='constructor_active'})
            manifest.update(model_timestep_seconds=float(env.model.opt.timestep),
                            forward_state_integration_max_abs_change=max(integration_deltas,default=0.),
                            primary_time_qpos_qvel_act_ctrl_preserved_exactly=True,
                            renderer_runtime=dependency_inventory())
        for name,value in [('RECONSTRUCTED_FRAMES.npz',frames),('RECONSTRUCTED_DETAIL_FRAMES.npz',details)]:
            np.savez_compressed(out/name,frames=np.asarray(value,dtype=np.uint8))
        # Source identity is rechecked after rendering; no source file is written.
        if any(sha(episode/name)!=digest for name,digest in manifest['origin_files'].items()):
            raise ValueError('An original artifact changed during reconstruction')
        if any(sha(ROOT/name)!=digest for name,digest in frozen['sources'].items()):
            raise ValueError('A frozen source changed during reconstruction')
        manifest.update(status='complete',frames=len(states),terminal_frame_included=True,
                        output_sha256={name:sha(out/name) for name in ['RECONSTRUCTED_FRAMES.npz','RECONSTRUCTED_DETAIL_FRAMES.npz']})
        write_json(out/'MANIFEST.json',manifest)
        (out/'README_ZH.md').write_text('封存状态重建回放，不是原始录像。\n\n逐帧恢复完整 integration state 后只做 mj_forward 和渲染。实际物理积分为零；构造阶段被抑制的调用数量见 MANIFEST。派生量可能与原始画面相差一个积分步，不声称像素完全复现。原 cell 和 seal 没有修改。\n')
        return manifest
    finally:
        if env is not None:env.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--episode',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args();manifest=reconstruct(args.episode,args.out)
    print(f'Reconstructed {manifest["frames"]} sealed states; actual integration steps: {manifest["physics_steps_executed"]}')


if __name__=='__main__':main()
