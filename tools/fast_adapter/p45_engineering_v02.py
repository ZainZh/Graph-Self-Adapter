"""P45 preregistered physical and visible-observer engineering qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from prove.fast_adapter.robot_geometry_env_p45_v02 import (
    RobotGeometrySkill, configuration, GeometryTaskController, source_assets,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def annotated_video(out, frames, trace, task, result):
    """Presentation derivative; preserves sealed raw images and state traces."""
    from PIL import Image, ImageDraw, ImageFont
    import imageio.v2 as imageio
    font_path = next((p for p in [Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'), Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc'), Path('/usr/share/fonts/truetype/arphic/uming.ttc')] if p.exists()), None)
    font = ImageFont.truetype(str(font_path), 21) if font_path else ImageFont.load_default()
    stages = {'shared observation motion': '共同观察动作', 'raise hand to shared clearance': '抬升至共同安全高度', 'align above handle outside approach': '在把手外侧上方对齐', 'descend outside handle': '在把手外侧下降', 'align outside handle': '按估计几何对齐把手外侧', 'approach handle with open fingers': '张开夹爪接近把手', 'close physical fingers': '位置与朝向到位后闭合', 'hold insertion': '保持插入状态', 'align held peg': '调整已持物块的位置与朝向', 'approach slot': '靠近槽口', 'insert along estimated axis': '沿槽口方向插入', 'align outside cup handle': '移动到杯柄外侧', 'approach handle with open fingers': '张开夹爪接近杯柄', 'close physical Panda fingers': '真实手指闭合接触', 'lift with finger contacts': '保持接触并抬升'}
    title = 'Panda 七关节机械臂 | 工程 Oracle 资格样例（使用目标真值）'
    note = '插入：从开始即假设已牢固持物；物块与槽壁保持碰撞' if task == 'insert' else '杯柄抓取：自由物体，仅靠夹爪接触；无吸附或焊接'
    writer = imageio.get_writer(out / 'annotated.mp4', fps=round(1 / .06), quality=8, macro_block_size=16)
    try:
        for i, frame in enumerate(frames):
            canvas = Image.new('RGB', (896, 560), '#111b25')
            canvas.paste(Image.fromarray(frame), (0, 58))
            draw = ImageDraw.Draw(canvas)
            draw.text((14, 5), title, font=font, fill='white')
            draw.text((14, 30), note, font=font, fill='#bcd4df')
            stage = stages.get(trace[i]['stage'], trace[i]['stage'])
            draw.text((14, 512), f'{i * .06:4.1f}s  |  {stage}', font=font, fill='white')
            writer.append_data(np.asarray(canvas))
    finally:
        writer.close()


def run(task, seed, split, out, render=True, steps=240):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    cfg = configuration(task, seed, split)
    sources = ['prove/fast_adapter/robot_geometry_env_p45_v02.py', 'prove/fast_adapter/geometry3d_env.py', 'tools/fast_adapter/p45_engineering_v02.py', 'prove/fast_adapter/p45_handle_surfaces.py', 'prove/fast_adapter/p45_visible_frame.py']
    js(out / 'PRE_RUN.json', dict(scope='p45_engineering_oracle_package2', config=cfg, steps=steps, render=render, timestamp=time.time(), sources={p: sha(p) for p in sources}, assets=source_assets(), controller='cup bounded scratch-state IK with joint margin and common clearance path; insert package1 DLS; mass-matrix joint PD; 60ms/action', cup_base_z=-.35, cup_table_half_x=.35, canonical_target_frame_unchanged=True, goal_part_mask='cup visible physical front surfaces; insert rim', contact_material_and_success_thresholds_unchanged=True, insertion_secure_grasp_assumed=task == 'insert', cup_attachment=False, task_phase_control='estimated relationship and own SE3; no forced time-based closure', qualification_gates=dict(sustained_terminal_states=10,max_contact_penetration=.002,max_arm_limit_rad=.001,max_finger_limit_m=.0001,valid_prefix_frames=8,median_prefix_corner_rms_px=4.)))
    for source in sources:
        target = out / 'source' / source
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(Path(source).read_bytes())
    e = RobotGeometrySkill(cfg, render=render)
    trace = []; frames = []; observations = []
    controller = GeometryTaskController(task, e.start_pos, e.start_R, cfg.get('length', 0.))
    try:
        if render:
            from PIL import Image
            Image.fromarray(e.render_global()).save(out / 'start.png')
        js(out / 'RESET.json', dict(proprio=e.proprio()[0].tolist(), robot=e.robot_tracking_diagnostics()))
        for t in range(steps):
            est = e.oracle()
            p, R, closed, stage = controller.command(t, est, e.proprio())
            state = e.snapshot()
            trace.append(dict(t=t, state=state, command_p=p, command_R=R, closed=closed, stage=stage, proprio=e.proprio(), controller=controller.state_dict()))
            if render:
                obs, rgb, mask = e.observe()
                observations.append(obs)
                frames.append(np.concatenate([e.render_global(), e.render_detail()], axis=1))
                trace[-1]['visible_corners'] = obs['corners']
            e.step(p, R, closed)
            trace[-1]['tracking'] = e.robot_tracking_diagnostics()
        terminal = e.snapshot()
        np.save(out / 'TRACE.npy', np.array(trace, dtype=object), allow_pickle=True)
        np.save(out / 'TERMINAL.npy', terminal, allow_pickle=True)
        sealed = ['TRACE.npy', 'TERMINAL.npy']
        if render:
            np.savez_compressed(out / 'FRAMES.npz', frames=np.array(frames))
            np.save(out / 'OBS.npy', np.array(observations,dtype=object), allow_pickle=True)
            sealed.append('OBS.npy')
            sealed.append('FRAMES.npz')
            Image.fromarray(e.render_global()).save(out / 'end.png')
        js(out / 'SEAL.json', {name: sha(out / name) for name in sealed})
        facts = e.facts(); held = []; penetrations = []
        for state in [x['state'] for x in trace[-9:]] + [terminal]:
            e.restore(state)
            held.append(e.facts()['success'])
            penetrations.append(max([max(0., -float(c.dist)) for c in e.data.contact] + [0.]))
        facts.update(success=bool(all(held) and max(penetrations) < .002), last10_successes=sum(held), max_last10_penetration=max(penetrations))
        result = dict(task=task, seed=seed, split=split, scope='p45_engineering_oracle', **facts, robot=e.robot_tracking_diagnostics(), max_position_tracking_error=max((x['tracking']['position_error'] for x in trace), default=0.), max_angle_tracking_error=max((x['tracking']['angle_error'] for x in trace), default=0.), max_ik_position_error=max((x['tracking']['ik_position_error'] for x in trace), default=0.))
        result['max_arm_joint_limit_violation_rad'] = max((x['tracking']['max_substep_arm_limit_violation_rad'] for x in trace), default=0.)
        result['max_finger_joint_limit_violation_m'] = max((x['tracking']['max_substep_finger_limit_violation_m'] for x in trace), default=0.)
        result['final_controller_phase'] = controller.phase
        result['max_full_trajectory_contact_penetration'] = max((x['tracking']['max_substep_penetration_m'] for x in trace), default=0.)
        result['physical_qualified'] = bool(result['max_full_trajectory_contact_penetration'] < .002 and result['success'] and result['max_arm_joint_limit_violation_rad'] <= .001 and result['max_finger_joint_limit_violation_m'] <= .0001)
        if render:
            errors = []
            for row, obs in zip(trace[:12], observations[:12]):
                if obs['corners'] is None: continue
                e.restore(row['state'])
                center, target_R, half = e.oracle()
                if task == 'insert':
                    half = half + .006
                    center = center + target_R[:,2] * .0006
                corners = np.array([[-half[0],half[1],0],[half[0],half[1],0],[half[0],-half[1],0],[-half[0],-half[1],0]])@target_R.T+center
                cam = (corners-obs['camera_position'])@obs['camera_rotation']
                K = obs['K']
                uv = np.c_[K[0,2]+K[0,0]*cam[:,0]/(-cam[:,2]),K[1,2]-K[1,1]*cam[:,1]/(-cam[:,2])]
                errors.append(min(float(np.sqrt(np.square(np.roll(obs['corners'],k,axis=0)-uv).sum(-1).mean())) for k in range(4)))
            result['prefix_valid_observations'] = len(errors)
            result['prefix_corner_rms_px_median'] = float(np.median(errors)) if errors else None
            result['observer_qualified'] = bool(len(errors)>=8 and np.median(errors)<=4.)
            result['qualified'] = result['physical_qualified'] and result['observer_qualified']
            e.restore(terminal)
        else:
            result['observer_qualified'] = None
            result['qualified'] = False
        js(out / 'RESULT.json', result)
        if render:
            import imageio.v2 as imageio
            imageio.mimsave(out / 'raw.mp4', frames, fps=round(1 / .06), quality=8, macro_block_size=16)
            annotated_video(out, frames, trace, task, result)
        print(json.dumps(result), flush=True)
        return result
    finally:
        e.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--task', choices=['cup', 'insert'], required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--split', choices=['id', 'geometry'], default='id')
    p.add_argument('--out', required=True)
    p.add_argument('--steps', type=int, default=240)
    p.add_argument('--no-render', action='store_true')
    a = p.parse_args()
    run(a.task, a.seed, a.split, a.out, not a.no_render, a.steps)
