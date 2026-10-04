"""Post-hoc illustration of one sealed insertion rescue, with no new rollout."""
import json
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
import mujoco

from prove.fast_adapter.robot_geometry_env import RobotGeometrySkill
from tools.fast_adapter.p44_evaluate import restore_physical, sha
from tools.fast_adapter.p44_audit import check_seal
from tools.fast_adapter.p44_report import font, STAGES


def main():
    root = Path('evaluations/fast_adapter_p44_robot_geometry_v01')
    candidates = []
    for cell in sorted((root / 'development/insert_seed11').glob('geometry_*_continuous')):
        fixed = cell.with_name(cell.name.replace('_continuous', '_fixed'))
        if (cell / 'RESULT.json').exists() and (fixed / 'RESULT.json').exists():
            if json.loads((cell / 'RESULT.json').read_text())['success'] and not json.loads((fixed / 'RESULT.json').read_text())['success']:
                candidates.append((fixed, cell))
    if not candidates:
        raise RuntimeError('No model11 OOD rescue to illustrate')
    cells = candidates[0]
    output = root / 'summary/insert_rescue_reconstruction'
    output.mkdir(parents=True, exist_ok=False)
    original_step = mujoco.mj_step
    suppressed = [0]
    def skip_initialization(*args, **kwargs):
        suppressed[0] += 1
    def forbid_step(*args, **kwargs):
        raise RuntimeError('Physical execution forbidden in sealed-state media')
    envs, records, sources = [], [], []
    try:
        for cell in cells:
            pre, _ = check_seal(cell)
            for source, digest in pre['sources'].items():
                assert sha(source) == digest
            mujoco.mj_step = skip_initialization
            envs.append(RobotGeometrySkill(pre['cfg']))
            mujoco.mj_step = forbid_step
            trace = np.load(cell / 'TRACE.npy', allow_pickle=True)
            terminal = np.load(cell / 'TERMINAL.npy', allow_pickle=True).item()
            records.append((trace, terminal, json.loads((cell / 'RESULT.json').read_text())))
            sources.append(dict(cell=str(cell), sha256={n: sha(cell / n) for n in ['TRACE.npy', 'TERMINAL.npy', 'PRE_RUN.json', 'RESULT.json', 'SEAL.json']}))
        width, height, steps = 1792, 816, len(records[0][0])
        video = output / 'insert_rescue.mp4'
        encoder = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '15', '-i', '-', '-an', '-c:v', 'libx264', '-threads', '2', '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(video)], stdin=subprocess.PIPE)
        title, body, small = font(34), font(28), font(23)
        try:
            for t in range(steps + 1):
                canvas = Image.new('RGB', (width, height), '#111a26')
                draw = ImageDraw.Draw(canvas)
                draw.text((18, 10), 'Panda 插入：固定 z 失败，在线适配救回的开发个例', font=title, fill='white')
                draw.text((18, 56), f'条件 {records[0][2]["seed"]} · 模型 11 · 物块从开始已牢固持有 · 由封存状态重建，非原始 RGB', font=small, fill='#becddb')
                for j, (env, record) in enumerate(zip(envs, records)):
                    trace, terminal, result = record
                    state = trace[t] if t < steps else terminal
                    restore_physical(env, state)
                    x = j * 896
                    draw.text((x + 15, 100), '固定 z' if j == 0 else '在线更新 z', font=body, fill='#a8ebd6')
                    canvas.paste(Image.fromarray(env.render_global()), (x, 145))
                    canvas.paste(Image.fromarray(env.render_detail()), (x + 448, 145))
                    stage = trace[min(t, steps - 1)]['stage']
                    draw.text((x + 15, 613), f'执行步 {t}/{steps} · {STAGES.get(stage, stage)}', font=small, fill='white')
                    if t == steps:
                        draw.text((x + 15, 654), '最终判据：通过' if result['success'] else '最终判据：未通过', font=body, fill='#8ee3bf' if result['success'] else '#f2b3ab')
                        draw.text((x + 15, 697), f'深度 {1000*result["depth"]:.1f} mm · 横向误差 {1000*result["lateral"]:.1f} mm · 角差 {result["angle"]:.3f} rad', font=small, fill='#cfdae4')
                draw.text((18, 774), '事后按条件顺序选出的救回展示；未新增执行／神经推理，不改变全部测试统计；其余失败保留。', font=small, fill='#bac9d8')
                raw = np.asarray(canvas).tobytes()
                for _ in range(45 if t == steps else 1):
                    encoder.stdin.write(raw)
                if t == steps:
                    canvas.save(output / 'cover.jpg')
        finally:
            encoder.stdin.close()
        assert encoder.wait() == 0
        manifest = dict(scope='posthoc_development_rescue_illustration', selection='first sorted geometry condition with model11 fixed failure and continuous success', sources=sources,
                        video=str(video), video_sha256=sha(video), frames=steps+45,
                        original_rgb=False, physics_steps_executed=0, network_calls=0,
                        constructor_steps_suppressed=suppressed[0], generator_sha256=sha(__file__))
        (output / 'MANIFEST.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
        print(json.dumps(manifest), flush=True)
    finally:
        mujoco.mj_step = original_step
        for env in envs:
            env.close()


if __name__ == '__main__':
    main()
