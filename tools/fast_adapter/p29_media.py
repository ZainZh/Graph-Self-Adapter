"""A few annotated sealed-trajectory examples per task, never outcome evidence."""
import json
import argparse
import subprocess
from pathlib import Path
import numpy as np
import torch
import mujoco
from PIL import Image, ImageDraw, ImageFont
from tools.fast_adapter.p29_common import R, TASKS, SEEDS, BLOCK, HORIZON, SUPPORT, task_root, save_json, digest, ObjectEnv
from tools.fast_adapter.p29_eval import STRATA
from tools.fast_adapter.p29_report import arrays, TASK_NAMES as NAMES, STRATUM_NAMES


def render_state(env, raw, index, step):
    for k in ('qpos', 'qvel', 'mocap_pos', 'mocap_quat'):
        getattr(env.env.data, k)[:] = raw['states'][k][index, step].numpy()
    mujoco.mj_forward(env.env.model, env.env.data)
    return Image.fromarray(env.rgb())


def main(phase='final',round_name='r1'):
    out = R/'summary'/phase/round_name/'media'; out.mkdir(parents=True, exist_ok=False)
    tasks = [t for t in TASKS if all((task_root(t)/phase/round_name/st/'RESULT.json').exists() for st in STRATA)]
    selected = []
    for task in tasks:
        choices = {}; fallback = []
        # Explain the predeclared primary OOD setting first, then other strata.
        for st in ('geometry_ood', 'train', 'canonical'):
            result = json.loads((task_root(task)/phase/round_name/st/'RESULT.json').read_text())
            a = arrays(result)
            for si, seed in enumerate(SEEDS):
                correct = a['fast_correct'][si].astype(bool); base = a['base'].astype(bool)
                static = a['robot'][si].astype(bool)
                for kind, mask in [('rescue', correct & ~base & ~static), ('harm', base & ~correct)]:
                    ids = np.flatnonzero(mask)
                    if len(ids) and kind not in choices:
                        choices[kind] = (task, st, seed, int(ids[0]), kind)
                for kind, mask in [('success', correct), ('failure', ~correct)]:
                    ids = np.flatnonzero(mask)
                    if len(ids):
                        fallback.append((task, st, seed, int(ids[0]), kind))
        group = list(choices.values())
        for item in fallback:
            if len(group) >= 2:
                break
            if item[:4] not in [x[:4] for x in group]:
                group.append(item)
        selected.extend(group)
    font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 22)
    small = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc', 17)
    videos = []
    for task, st, seed, condition, kind in selected:
        folder = task_root(task)/phase/round_name/st
        offset = condition//BLOCK*BLOCK; j = condition % BLOCK
        arms = {'base': '基座／不更新z', f'robot_seed{seed}': '仅末端反馈适配',
                f'history_seed{seed}': '同物体反馈History', f'fast_correct_seed{seed}': '物体反馈梯度适配'}
        paths = {name: folder/f'{name}_{offset:03d}.pt' for name in arms}
        raw = {k: torch.load(p, weights_only=True) for k, p in paths.items()}
        metrics = {k: json.loads(p.with_suffix('.json').read_text()) for k, p in paths.items()}
        sp = folder/f'SUPPORT_correct_{offset:03d}.pt'; support = torch.load(sp, weights_only=True)
        cfg = metrics['base']['configs'][j]
        env = ObjectEnv(cfg); env.reset(support['reset_seeds'][j])
        file = out/f'{task}_{st}_{kind}_s{seed}_c{condition}.mp4'
        width, height = 1344, 600
        proc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo',
            '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', '12', '-i', '-',
            '-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p', str(file)], stdin=subprocess.PIPE)
        for t in range(SUPPORT+1):
            image = Image.new('RGB', (width, height), (22, 31, 38)); draw = ImageDraw.Draw(image)
            draw.text((14, 10), NAMES[task]+'｜'+STRATUM_NAMES[st]+f'｜共同支持采集 {t}/160', font=font, fill='white')
            draw.text((14, 45), '读取局部图像与本体反馈；抓取的第二对象为空。执行正常，无机械臂动作扰动。', font=small, fill=(195, 217, 230))
            image.paste(render_state(env, support, j, t).resize((448, 448)), (448, 95))
            draw.text((20, 180), '采集动作→物体变化反馈', font=font, fill=(162, 227, 205))
            draw.text((20, 224), '随后仅做3次z梯度更新', font=font, fill='white')
            draw.text((20, 268), '160次动作 ≠ 3次更新', font=font, fill='white')
            draw.text((935, 180), '所有模型权重固定', font=font, fill='white')
            draw.text((935, 224), '支持与测试使用不同初态', font=font, fill='white')
            draw.text((935, 268), '同一个物体，更换初态', font=font, fill='white')
            proc.stdin.write(np.asarray(image).tobytes())
        env.close()
        envs = {arm: ObjectEnv(cfg) for arm in arms}
        for arm, e in envs.items():
            e.reset(raw[arm]['reset_seeds'][j])
        for t in range(HORIZON+1):
            image = Image.new('RGB', (width, height), (22, 31, 38)); draw = ImageDraw.Draw(image)
            stage_name = '独立测试' if phase == 'final' else '开发对照'
            draw.text((14, 10), NAMES[task]+'｜'+STRATUM_NAMES[st]+f'｜{stage_name} {t}/200', font=font, fill='white')
            draw.text((14, 45), '本阶段z和权重固定；四列读取相同初态。下方注释来自事后接触评价，不供模型读取。', font=small, fill=(195, 217, 230))
            for col, (arm, label) in enumerate(arms.items()):
                x = col*336
                draw.text((x+10, 82), label, font=font, fill=(162, 227, 205))
                image.paste(render_state(envs[arm], raw[arm], j, t).resize((336, 336)), (x, 120))
                if t:
                    info = metrics[arm]['metrics'][j][t-1]
                    if task == 'mug-handle-lift':
                        stage = '握住把手并稳定抬升' if info['success'] else ('握住把手，抬升／保持中' if info['bilateral_handle'] else '接近／尝试夹住把手')
                    else:
                        stage = '工具作用使木块位移达标' if info['success'] else ('锤头正在作用于木块' if info['head_target_contact'] else '持锤向目标移动' if info['bilateral_handle'] else '接近／尝试抓住锤柄')
                    draw.text((x+10, 466), stage, font=small, fill='white')
                    any_so_far = any(v['success'] for v in metrics[arm]['metrics'][j][:t])
                    draw.text((x+10, 496), '本回合'+('曾达标' if any_so_far else '尚未达标'), font=small, fill=(195, 217, 230))
                if t == HORIZON:
                    ok = metrics[arm]['success'][j]; final = metrics[arm]['terminal_success'][j]
                    draw.text((x+10, 529), '任务回合判据：'+('成功' if ok else '失败'), font=font,
                              fill=(120, 230, 165) if ok else (255, 160, 125))
                    draw.text((x+10, 562), '结束时：'+('仍达标' if final else '当前未达标'), font=small,
                              fill=(120, 230, 165) if final else (255, 160, 125))
            proc.stdin.write(np.asarray(image).tobytes())
            if t == HORIZON:
                image.save(file.with_suffix('.png'))
        proc.stdin.close(); assert proc.wait() == 0
        for e in envs.values():
            e.close()
        videos.append({'task': task, 'stratum': st, 'seed': seed, 'condition': condition,
                       'kind': kind, 'file': str(file), 'sha256': digest(file), 'frames': SUPPORT+HORIZON+2,
                       'sources': [{'path': str(p), 'sha256': digest(p)} for p in [sp, *paths.values()]],
                       'outcomes': {k: {'success_any': m['success'][j], 'terminal_success': m['terminal_success'][j]} for k, m in metrics.items()},
                       'scope': 'Post-seal state re-render, no new physical rollout. Selected descriptive cases, not rate evidence.'})
        print('VIDEO_COMPLETE', str(file), flush=True)
    save_json(out/'VIDEO_INDEX.json', {'selection': 'Per task, first geometryOOD/ID/canonical case rescued relative to both base and robot-only feedback, first base harm; fallback first success/failure. At most2 per task. Display selection is not rate evidence.', 'videos': videos})


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',default='final',choices=['final','development']);p.add_argument('--round',default='r1');a=p.parse_args();main(a.phase,a.round)
