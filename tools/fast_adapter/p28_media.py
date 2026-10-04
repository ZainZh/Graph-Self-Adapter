"""A few annotated sealed-trajectory examples per task, never outcome evidence."""
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
import mujoco
from PIL import Image, ImageDraw, ImageFont
from tools.fast_adapter.p28_common import R, TASKS, SEEDS, task_root, save_json, digest, TaskEnv
from tools.fast_adapter.p28_eval import EVAL_STRATA
from tools.fast_adapter.p28_report import arrays, NAMES, STRATUM_NAMES


def render_state(env, raw, index, step):
    for k in ('qpos', 'qvel', 'mocap_pos', 'mocap_quat'):
        getattr(env.env.data, k)[:] = raw['states'][k][index, step].numpy()
    mujoco.mj_forward(env.env.model, env.env.data)
    return Image.fromarray(env.rgb())


def main():
    out = R/'summary/media'; out.mkdir(parents=True, exist_ok=False)
    done = json.loads((R/'CAMPAIGN_DONE.json').read_text())
    selected = []
    for task in done['qualified_tasks']:
        choices = {}; fallback = []
        # Explain the predeclared primary OOD setting first, then other strata.
        for st in ('response_ood', 'response_id', 'native'):
            result = json.loads((task_root(task)/'evaluation'/st/'RESULT.json').read_text())
            a = arrays(result)
            for si, seed in enumerate(SEEDS):
                correct = a['fast_correct'][si].astype(bool); base = a['base'].astype(bool)
                static = a['fast_mean'][si].astype(bool)
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
        folder = task_root(task)/'evaluation'/st
        offset = condition//16*16; j = condition % 16
        arms = {'base': '基座／不更新z', f'fast_mean_seed{seed}': '固定训练平均z',
                f'fast_donor_seed{seed}': '错配反馈z', f'fast_correct_seed{seed}': '本环境梯度更新z'}
        paths = {name: folder/f'{name}_{offset:03d}.pt' for name in arms}
        raw = {k: torch.load(p, weights_only=True) for k, p in paths.items()}
        metrics = {k: json.loads(p.with_suffix('.json').read_text()) for k, p in paths.items()}
        sp = folder/f'SUPPORT_correct_{offset:03d}.pt'; support = torch.load(sp, weights_only=True)
        env = TaskEnv(task); env.reset(support['seeds'][j])
        file = out/f'{task}_{st}_{kind}_s{seed}_c{condition}.mp4'
        width, height = 1344, 600
        proc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo',
            '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', '12', '-i', '-',
            '-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p', str(file)], stdin=subprocess.PIPE)
        for t in range(81):
            image = Image.new('RGB', (width, height), (22, 31, 38)); draw = ImageDraw.Draw(image)
            draw.text((14, 10), NAMES[task]+'｜'+STRATUM_NAMES[st]+f'｜共同支持采集 {t}/80', font=font, fill='white')
            draw.text((14, 45), '固定基座先执行80个动作；模型只读取RGB、本体状态和自己的动作，不读取目标位置或成功标签。', font=small, fill=(195, 217, 230))
            image.paste(render_state(env, support, j, t).resize((448, 448)), (448, 95))
            draw.text((20, 180), '采集实际动作→位移反馈', font=font, fill=(162, 227, 205))
            draw.text((20, 224), '随后仅做3次z梯度更新', font=font, fill='white')
            draw.text((20, 268), '80次动作 ≠ 3次更新', font=font, fill='white')
            draw.text((935, 180), '所有模型权重固定', font=font, fill='white')
            draw.text((935, 224), '支持与测试使用不同初态', font=font, fill='white')
            draw.text((935, 268), '同一隐藏动作响应条件', font=font, fill='white')
            proc.stdin.write(np.asarray(image).tobytes())
        env.close()
        envs = {arm: TaskEnv(task) for arm in arms}
        for arm, e in envs.items():
            e.reset(raw[arm]['seeds'][j])
        for t in range(161):
            image = Image.new('RGB', (width, height), (22, 31, 38)); draw = ImageDraw.Draw(image)
            draw.text((14, 10), NAMES[task]+'｜'+STRATUM_NAMES[st]+f'｜独立初态执行 {t}/160', font=font, fill='white')
            draw.text((14, 45), '本阶段z固定；后三列为同一个动作头，只改变z来源。下方任务状态是事后评价，不供控制模型读取。', font=small, fill=(195, 217, 230))
            for col, (arm, label) in enumerate(arms.items()):
                x = col*336
                draw.text((x+10, 82), label, font=font, fill=(162, 227, 205))
                image.paste(render_state(envs[arm], raw[arm], j, t).resize((336, 336)), (x, 120))
                if t:
                    info = metrics[arm]['metrics'][j][t-1]
                    if task == 'pick-place-v3':
                        stage = '当前到达目标范围' if info['success'] else ('已抓持，移动中' if info.get('grasp_success') else '接近／抓取／调整')
                    else:
                        stage = '当前抽屉已打开' if info['success'] else '接近把手／接触拉动'
                    draw.text((x+10, 466), stage, font=small, fill='white')
                    any_so_far = any(v['success'] for v in metrics[arm]['metrics'][j][:t])
                    draw.text((x+10, 496), '本回合'+('曾达标' if any_so_far else '尚未达标'), font=small, fill=(195, 217, 230))
                if t == 160:
                    ok = metrics[arm]['success'][j]; final = metrics[arm]['terminal_success'][j]
                    draw.text((x+10, 529), '官方回合判据：'+('成功' if ok else '失败'), font=font,
                              fill=(120, 230, 165) if ok else (255, 160, 125))
                    draw.text((x+10, 562), '结束时：'+('仍达标' if final else '未保持达标'), font=small,
                              fill=(120, 230, 165) if final else (255, 160, 125))
            proc.stdin.write(np.asarray(image).tobytes())
            if t == 160:
                image.save(file.with_suffix('.png'))
        proc.stdin.close(); assert proc.wait() == 0
        for e in envs.values():
            e.close()
        videos.append({'task': task, 'stratum': st, 'seed': seed, 'condition': condition,
                       'kind': kind, 'file': str(file), 'sha256': digest(file), 'frames': 242,
                       'sources': [{'path': str(p), 'sha256': digest(p)} for p in [sp, *paths.values()]],
                       'outcomes': {k: {'success_any': m['success'][j], 'terminal_success': m['terminal_success'][j]} for k, m in metrics.items()},
                       'scope': 'Post-seal state re-render, no new physical rollout. Selected descriptive cases, not rate evidence.'})
        print('VIDEO_COMPLETE', str(file), flush=True)
    save_json(out/'VIDEO_INDEX.json', {'selection': 'Per task, first OOD/ID/native case rescued relative to both base and static mean, first base harm; fallback first success/failure. At most2 per task.', 'videos': videos})


if __name__ == '__main__':
    main()
