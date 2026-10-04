"""Annotated, post-seal state re-render. No new task rollout or policy call."""
import json
import subprocess
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from tools.fast_adapter.p30v2_common import ROOT, TASKS, dump, digest
from tools.fast_adapter.p30v2_report import ARMS, LABELS, TASK_NAMES, SPLITS
from prove.fast_adapter.object_interaction_env import ObjectEnv, mujoco


def main():
    out = ROOT/'summary'/'media'; out.mkdir(exist_ok=False)
    fonts = '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    title, font = ImageFont.truetype(fonts, 23), ImageFont.truetype(fonts, 17)
    videos = []
    for task in TASKS:
        choices, fallback = {}, []
        for split in ('geometry_ood', 'train'):
            for seed in (11, 22):
                folder = ROOT/task/'development'/f'seed{seed}'
                rows = json.loads((folder/'RESULT.json').read_text())['rows']
                rr = {(r['condition'], r['arm']): r for r in rows if r['split'] == split}
                for condition in range(16):
                    b, g = rr[(condition, 'base')]['success'], rr[(condition, 'gradient')]['success']
                    item = (split, seed, condition)
                    if g and not b: choices.setdefault('rescue', item)
                    if b and not g: choices.setdefault('harm', item)
                    fallback.append(('success' if g else 'failure', item))
        selections = list(choices.items())
        for item in fallback:
            if len(selections) >= 2: break
            if item[1] not in [v for _, v in selections]: selections.append(item)
        for kind, (split, seed, condition) in selections:
            folder = ROOT/task/'development'/f'seed{seed}'
            j, offset = condition % 8, condition // 8 * 8
            paths = {a: folder/f'{split}_{offset:03d}_{a}.pt' for a in ARMS}
            raw = {a: torch.load(p, weights_only=True) for a, p in paths.items()}
            meta = {a: json.loads(p.with_suffix('.json').read_text()) for a, p in paths.items()}
            cfg = meta['base']['rows'][j]['config']
            env = ObjectEnv(cfg); env.reset(raw['base']['reset_seeds'][j])
            file = out/f'{task}_{split}_{kind}_s{seed}_c{condition}.mp4'
            width, height = 1400, 564
            proc = subprocess.Popen(['ffmpeg', '-loglevel', 'error', '-y', '-f', 'rawvideo',
                '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', '12',
                '-i', '-', '-c:v', 'libx264', '-crf', '23', '-pix_fmt', 'yuv420p',
                '-movflags', '+faststart', str(file)], stdin=subprocess.PIPE)
            try:
                for t in range(201):
                    image = Image.new('RGB', (width, height), (22, 31, 38)); draw = ImageDraw.Draw(image)
                    draw.text((14, 10), TASK_NAMES[task]+'｜'+SPLITS[split]+f'｜开发对照：执行步 {t}/200｜配置{condition}，种子{seed}', font=title, fill='white')
                    draw.text((14, 48), '五列初态配对；模型权重始终冻结。注释来自封存后的接触评价，不供控制器读取；按已保存状态重绘。', font=font, fill=(190, 210, 223))
                    for col, arm in enumerate(ARMS):
                        x = col*280
                        draw.text((x+9, 88), LABELS[arm], font=font, fill=(153, 226, 204))
                        for k, states in raw[arm]['states'].items(): getattr(env.env.data, k)[:] = states[j, t].numpy()
                        mujoco.mj_forward(env.env.model, env.env.data)
                        image.paste(Image.fromarray(env.rgb()).resize((280, 280)), (x, 120))
                        count = 0 if arm in ('base', 'fixed_mean') else min(t, 3) if arm == 'first3' else t
                        update = '无在线更新' if arm in ('base', 'fixed_mean') else f'{"History状态" if arm == "history" else "z梯度"}更新 {count} 次'
                        draw.text((x+9, 410), update, font=font, fill='white')
                        if t:
                            m = meta[arm]['metrics'][j][t-1]
                            if task == TASKS[0]:
                                stage = '抓持＋抬升保持达标' if m['success'] else '夹住把手，抬升/保持' if m['bilateral_handle'] else '尚未双侧夹住把手'
                            else:
                                stage = '工具作用＋位移达标' if m['success'] else '锤头正接触木块' if m['head_target_contact'] else '持锤移动/对齐' if m['bilateral_handle'] else '当前未双侧握住锤柄'
                            draw.text((x+9, 442), stage, font=font, fill='white')
                            done = any(v['success'] for v in meta[arm]['metrics'][j][:t])
                            draw.text((x+9, 474), '本回合'+('已达标' if done else '尚未达标'), font=font, fill=(185, 206, 222))
                        else:
                            draw.text((x+9, 442), '已持工具，开始交互' if task == TASKS[1] else '开始抓取', font=font, fill='white')
                        if t == 200:
                            row = meta[arm]['rows'][j]
                            draw.text((x+9, 510), ('回合成功' if row['success'] else '回合失败')+'；终态'+('达标' if row['terminal_success'] else '未达标'), font=font, fill=(122, 230, 168) if row['success'] else (255, 166, 131))
                    proc.stdin.write(np.asarray(image).tobytes())
                    if t == 200: image.save(file.with_suffix('.png'))
                proc.stdin.close(); assert proc.wait() == 0
            finally:
                env.close()
                if proc.poll() is None: proc.terminate(); proc.wait()
            videos.append({'task': task, 'split': split, 'seed': seed, 'condition': condition,
                'kind': kind, 'file': str(file), 'sha256': digest(file), 'frames': 201,
                'sources': [{'path': str(p), 'sha256': digest(p)} for p in
                            [q for p in paths.values() for q in (p, p.with_suffix('.json'))]],
                'outcomes': {a: {k: meta[a]['rows'][j][k] for k in ('success', 'terminal_success')} for a in ARMS}})
            print('VIDEO_COMPLETE', str(file), flush=True)
    dump(out/'VIDEO_INDEX.json', {'scope': 'Post-seal descriptive display; no new task rollout; not rate evidence.',
        'selection': 'At most two per task: first gradient rescue and first harm versus base (OOD then ID, seed11 then22, ascending condition); fill first other gradient success/failure if needed.',
        'annotations': 'Control step, cumulative updates, contact/holding stage, success so far, final episode and terminal outcomes. Metrics are evaluator-only.',
        'videos': videos})


if __name__ == '__main__':
    main()
