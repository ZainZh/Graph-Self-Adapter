#!/usr/bin/env python3
"""Present three sealed P54/P55 episodes; never step or restore a simulator.

Code/prefix images are drawn from sealed qpos using mj_fwdPosition only.
Short-skill images are the RGB saved by the episode during execution.
"""
import argparse
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tools.task_graph.p52_render_full_online import choose_font, stage_of

CELLS = (
    ('旧图 · 固定 z', 'fast_adapter_p54_book_scale_v01', 'cell_22_nominal_fixed_s90'),
    ('新图 · 固定 z', 'fast_adapter_p55_graph_approach_v01', 'cell_22_nominal_fixed_s90'),
    ('新图 · 在线 z', 'fast_adapter_p55_graph_approach_v01', 'cell_23_nominal_online_s90'),
)
STAGE = {'s7': '代码对齐', 's8': '粗接近', 's8a': '新增接近', 's9': '冻结短技能'}
STAGE.update({'s1': '移动', 's2': '接近', 's3': '抓取', 's4': '抬起', 's5': '搬运', 's6': '对齐'})
DESCRIPTION = {'s7': '公开本体反馈对齐', 's8': '代码接近盒口',
               's8a': '沿公开参考轴继续接近', 's9': '短技能局部对齐与放入'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def verified(episode, seal, relative, expected=None):
    path = episode / relative
    digest = sha(path)
    if digest != seal['files'].get(relative) or (expected is not None and digest != expected):
        raise ValueError(f'Unsealed or changed source: {path}')
    return {'file': relative, 'sha256': digest}


def display_binding(binding):
    """Resolve the sealed parent chain and apply the P54 book scale to its XML."""
    chain = []
    current = binding
    while 'environment' not in current:
        pointer = current['parent']
        path = Path(pointer['path'])
        digest = sha(path)
        if digest != pointer['sha256']:
            raise ValueError('Parent binding changed: ' + str(path))
        chain.append({'file': str(path), 'sha256': digest})
        current = json.loads(path.read_text())
    from tools.task_graph.p54_book_geometry import scaled_xml
    p54 = next(x for x in [binding] + [json.loads(Path(p['file']).read_text()) for p in chain]
               if 'geometry' in x and 'properties' in x['geometry'])
    xml = scaled_xml(current['environment']['relocated_xml'], .9, p54['geometry']['properties'])
    return xml, current['public']['cameras'], chain


def load_episode(base, title, project, cell):
    episode = base / project / 'full_eval_v01' / cell / 'episode'
    result = json.loads((episode / 'RESULT.json').read_text())
    binding = json.loads((episode / 'BINDING.json').read_text())
    seal = json.loads((episode / 'SEAL.json').read_text())
    if result['cell']['scale'] != .9 or result['cell']['condition'] != 'nominal':
        raise ValueError('Expected nominal 90% book')
    if sha(episode / 'SEAL.json') != result['seal_sha256'] or sha(episode / 'BINDING.json') != seal['binding_sha256']:
        raise ValueError('Result/seal/binding mismatch')
    if not result['network_weights_unchanged'] or result['prefix_actions'] != 341:
        raise ValueError('Unexpected episode contract')
    if project.startswith('fast_adapter_p54'):
        if (result['code_actions'], result['actions'], result['graph_complete'], result['official_success']) != (120, 0, False, False):
            raise ValueError('Old graph outcome changed')
    elif cell.startswith('cell_22'):
        if (result['code_actions'], result['actions'], result['graph_complete'], result['official_success'], result['updates']) != (52, 17, True, True, 0):
            raise ValueError('Fixed-z outcome changed')
    elif (result['code_actions'], result['actions'], result['graph_complete'], result['official_success'], result['updates']) != (52, 24, False, False, 24):
        raise ValueError('Online-z outcome changed')
    bound = {}
    for relative in ('prefix/TRACE.jsonl', 'code/TRACE.jsonl', 'STAGE_TRANSITIONS.json', 'CODE_RESULT.json'):
        bound[relative] = verified(episode, seal, relative)
    prefix = rows(episode / 'prefix/TRACE.jsonl')
    code = rows(episode / 'code/TRACE.jsonl')
    micro_path = episode / 'public/ACTIONS.jsonl'
    micro = rows(micro_path) if micro_path.exists() else []
    if micro:
        bound['public/ACTIONS.jsonl'] = verified(episode, seal, 'public/ACTIONS.jsonl')
    if len(prefix) != 342 or len(code) != result['code_actions'] + 1 or len(micro) != result['actions'] + (1 if result['actions'] else 0):
        raise ValueError('Trace lengths differ from result')
    if prefix[-1].get('action') is not None or code[-1].get('action') is not None:
        raise ValueError('Missing terminal handoff observation')
    if micro and (micro[-1]['executed'] or any(not x['executed'] for x in micro[:-1])):
        raise ValueError('Invalid short-skill terminal observation')
    xml, cameras, parent_chain = display_binding(binding)
    timeline = []
    for folder, sequence, key in (('prefix', prefix[:-1], 'file'), ('code', code[:-1] if micro else code, 'file'), ('public', micro, 'observation')):
        for i, row in enumerate(sequence):
            relative = f'{folder}/{row[key]}'
            bound[relative] = verified(episode, seal, relative, row['sha256'])
            stage = stage_of(row) if folder == 'prefix' else (row['stage_id'] if folder == 'code' else 's9')
            timeline.append({'segment': folder, 'index': i, 'stage': stage,
                             'source': relative, 'sha256': bound[relative]['sha256'],
                             'executed': bool(row['executed']) if folder == 'public' else bool(row.get('action') is not None),
                             'updates': row.get('updates', 0),
                             'terminal': (folder == 'code' and not micro and i == len(code)-1) or
                                         (folder == 'public' and i == len(micro)-1)})
    if len(timeline) != result['total_actions'] + 1:
        raise ValueError('Timeline must show all actions and one terminal state')
    return dict(title=title, project=project, cell=cell, path=episode, result=result, binding=binding,
                xml=xml, cameras=cameras, parent_chain=parent_chain,
                seal=seal, bound=bound, timeline=timeline, result_sha256=sha(episode / 'RESULT.json'),
                seal_sha256=sha(episode / 'SEAL.json'))


def source_summary(ep):
    r = ep['result']
    return {'project': ep['project'], 'cell': ep['cell'], 'episode': str(ep['path']),
            'result_sha256': ep['result_sha256'], 'seal_sha256': ep['seal_sha256'],
            'source_bindings': ep['bound'], 'parent_binding_chain': ep['parent_chain'],
            'prefix_actions': r['prefix_actions'],
            'code_actions': r['code_actions'], 'short_skill_actions': r['actions'],
            'z_updates': r['updates'], 'official_success': r['official_success'],
            'graph_complete': r['graph_complete']}


def rgb_for(ep, item, render_context):
    path = ep['path'] / item['source']
    if item['segment'] == 'public':
        with np.load(path, allow_pickle=False) as saved:
            return saved['rgb'].copy()
    model, data, renderer, option = render_context
    with np.load(path, allow_pickle=False) as saved:
        data.qpos[:] = saved['supervisor_qpos']
    import mujoco
    mujoco.mj_fwdPosition(model, data)
    output = []
    for camera in ep['cameras']:
        renderer.update_scene(data, camera=camera, scene_option=option)
        output.append(renderer.render().copy())
    return output


def context(ep):
    import mujoco
    model = mujoco.MjModel.from_xml_string(ep['xml'])
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=448, width=448)
    option = mujoco.MjvOption()
    option.geomgroup[0] = 0
    option.geomgroup[1] = 1
    return model, data, renderer, option


def encoder(path, size, fps):
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
               '-s', f'{size[0]}x{size[1]}', '-r', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
               '-preset', 'fast', '-crf', '22', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(path)]
    return subprocess.Popen(command, stdin=subprocess.PIPE)


def send(process, picture, repeat=1):
    raw = np.asarray(picture, dtype=np.uint8).tobytes()
    for _ in range(repeat):
        process.stdin.write(raw)


def close(process):
    process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg failed')


def result_line(ep):
    r = ep['result']
    return f'终态：官方成功={"是" if r["official_success"] else "否"}；图完成={"是" if r["graph_complete"] else "否"}'


def comparison_frame(arms, views, positions, font, terminal_hold=False):
    canvas = Image.new('RGB', (1152, 990), (17, 23, 34))
    d = ImageDraw.Draw(canvas)
    title = ImageFont.truetype(str(font), 27)
    body = ImageFont.truetype(str(font), 20)
    small = ImageFont.truetype(str(font), 17)
    d.text((18, 7), '90% 书本 · 同名义初态 · 三臂 Code 起点对齐', font=title, fill='white')
    d.text((18, 48), '每个封存观察帧重复4次慢放；列内动作顺序不变；结束后定格', font=small, fill=(215, 220, 228))
    for column, (ep, rgb, position) in enumerate(zip(arms, views, positions)):
        x = column * 384
        item, frozen = position
        d.rectangle((x+3, 79, x+380, 969), outline=(70, 84, 101), width=2)
        d.text((x+12, 86), ep['title'], font=body, fill=(130, 220, 255))
        stage = item['stage']
        subtitle = f'{stage} {STAGE.get(stage, "")}  ·  {DESCRIPTION.get(stage, "")}'
        d.text((x+12, 118), subtitle, font=small, fill='white')
        for n, label in enumerate(('外部视角', '腕部视角')):
            tile = Image.fromarray(np.asarray(rgb[n], dtype=np.uint8)).resize((366, 366), Image.Resampling.LANCZOS)
            canvas.paste(tile, (x+9, 151+n*376))
            d.rectangle((x+9, 151+n*376, x+100, 176+n*376), fill=(20,25,32))
            d.text((x+15, 153+n*376), label, font=small, fill='white')
        d.text((x+12, 908), f'本列阶段动作 {item["index"]+1 if item["executed"] else item["index"]}  |  z更新 {item["updates"]}', font=small, fill=(220,225,233))
        if frozen:
            d.text((x+12, 934), result_line(ep), font=small, fill=(126,230,156) if ep['result']['official_success'] else (255,176,132))
            d.text((x+12, 960), '终态定格，无新增动作', font=small, fill=(255,204,112))
        elif item['terminal']:
            d.text((x+12, 934), result_line(ep), font=small, fill=(126,230,156) if ep['result']['official_success'] else (255,176,132))
        else:
            d.text((x+12, 934), '封存轨迹回放', font=small, fill=(210,216,228))
    if terminal_hold:
        d.text((12, 970), '均已结束；三列为封存终态，不代表同时完成', font=small, fill=(255,204,112))
    return canvas


def full_frame(ep, item, rgb, font, replay=False):
    canvas = Image.new('RGB', (896, 736), (17, 23, 34))
    d = ImageDraw.Draw(canvas)
    title = ImageFont.truetype(str(font), 28)
    body = ImageFont.truetype(str(font), 21)
    small = ImageFont.truetype(str(font), 18)
    d.text((18, 9), 'P55 · 90% 书本 · 新图固定 z · 原初态完整回合', font=title, fill='white')
    stage = item['stage']
    description = DESCRIPTION.get(stage, '原图抓取、抬起、搬运')
    d.text((18, 54), f'{stage} {STAGE.get(stage, "原图前缀")}  |  {description}', font=body, fill=(130,220,255))
    d.text((18, 91), '慢放重复封存帧，无新增动作' if replay else '原始动作顺序；全部动作状态与终态', font=small, fill=(220,225,233))
    for i, label in enumerate(('外部视角 agentview', '腕部视角')):
        canvas.paste(Image.fromarray(np.asarray(rgb[i], dtype=np.uint8)), (448*i, 126))
        d.rectangle((448*i, 126, 448*i+220, 155), fill=(20,25,32))
        d.text((448*i+9, 130), label, font=small, fill='white')
    for i, (name, active) in enumerate((('原图前缀', item['segment']=='prefix'), ('s7 对齐',stage=='s7'),
                                        ('s8 粗接近',stage=='s8'), ('s8a 新增接近',stage=='s8a'), ('s9 短技能',stage=='s9'))):
        x=6+i*178
        d.rectangle((x,583,x+172,626),fill=(48,145,220) if active else (61,68,80))
        d.text((x+9,592),name,font=small,fill='white')
    d.text((18,638), f'来源 {item["segment"]}/{item["index"]}  |  z更新 {item["updates"]}  |  网络权重冻结',font=small,fill=(220,225,233))
    d.text((18,670),'前缀/Code：封存位置重绘；短技能：执行时保存的 RGB',font=small,fill=(211,216,223))
    if item['terminal']:
        d.text((18,701),result_line(ep)+'；持书连续3帧判据',font=small,fill=(126,230,156))
    else:
        d.text((18,701),'结果只在实际终态显示；此单回合不证明在线 z 收益',font=small,fill=(255,204,112))
    return canvas


def render_comparison(arms, output, font, fps):
    contexts = [context(ep) for ep in arms]
    process = encoder(output / 'comparison_code_aligned.mp4', (1152,990), fps)
    timeline=[]
    lengths=[len(ep['timeline'])-ep['result']['prefix_actions'] for ep in arms]
    last_views=[None]*3
    cover=None
    frame=0
    try:
        for t in range(max(lengths)):
            positions=[]
            for i,ep in enumerate(arms):
                j=min(t,lengths[i]-1)
                item=ep['timeline'][ep['result']['prefix_actions']+j]
                frozen=t>=lengths[i]
                if not frozen:
                    last_views[i]=rgb_for(ep,item,contexts[i])
                positions.append((item,frozen))
            picture=comparison_frame(arms,last_views,positions,font)
            if cover is None:
                cover=picture.copy();cover.save(output/'comparison_cover.png')
            send(process,picture,4)
            timeline.append({'video_frame_start':frame,'video_frame_end':frame+3,'alignment_index':t,
                             'arms':[{'cell':ep['cell'],'source':item['source'],'source_sha256':item['sha256'],
                                      'stage':item['stage'],'terminal':item['terminal'],'frozen':frozen,
                                      'z_updates':item['updates']} for ep,(item,frozen) in zip(arms,positions)]})
            frame+=4
        for _ in range(2*fps):
            send(process,picture)
        frame+=2*fps
        timeline.append({'video_frame_start':frame-2*fps,'video_frame_end':frame-1,'terminal_hold':True,
                         'arms':[{'cell':ep['cell'],'source':ep['timeline'][-1]['source'],
                                  'source_sha256':ep['timeline'][-1]['sha256']} for ep in arms]})
        close(process)
    finally:
        for ctx in contexts:ctx[2].close()
        if process.poll() is None:process.kill();process.wait()
    return frame,timeline


def render_full(ep,output,font,fps):
    ctx=context(ep)
    process=encoder(output/'full_fixed_success.mp4',(896,736),fps)
    timeline=[];replay=[];frame=0
    try:
        for item in ep['timeline']:
            rgb=rgb_for(ep,item,ctx)
            picture=full_frame(ep,item,rgb,font)
            if frame==0:picture.save(output/'full_cover.png')
            send(process,picture)
            timeline.append({'video_frame_start':frame,'video_frame_end':frame,'source':item['source'],
                             'source_sha256':item['sha256'],'stage':item['stage'],'terminal':item['terminal'],
                             'z_updates':item['updates'],'replay':False})
            frame+=1
            if item['stage'] in ('s8a','s9'):replay.append((item,rgb))
        for item,rgb in replay:
            picture=full_frame(ep,item,rgb,font,replay=True)
            send(process,picture,4)
            timeline.append({'video_frame_start':frame,'video_frame_end':frame+3,'source':item['source'],
                             'source_sha256':item['sha256'],'stage':item['stage'],'terminal':item['terminal'],
                             'z_updates':item['updates'],'replay':True})
            frame+=4
        for _ in range(2*fps):send(process,picture)
        timeline.append({'video_frame_start':frame,'video_frame_end':frame+2*fps-1,
                         'terminal_hold':True,'source':ep['timeline'][-1]['source'],
                         'source_sha256':ep['timeline'][-1]['sha256']})
        frame+=2*fps
        close(process)
    finally:
        ctx[2].close()
        if process.poll() is None:process.kill();process.wait()
    return frame,timeline


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--evaluations-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path)
    p.add_argument('--check-metadata-only',action='store_true')
    p.add_argument('--font',type=Path)
    p.add_argument('--fps',type=int,default=20)
    args=p.parse_args()
    if args.fps<=0:raise ValueError('fps must be positive')
    arms=[load_episode(args.evaluations_root,*cell) for cell in CELLS]
    # The shared prefix and new-graph Code handoff must truly be the same source states.
    if len({x['timeline'][0]['sha256'] for x in arms})!=1:
        raise ValueError('Initial state differs across arms')
    if arms[1]['timeline'][341:393] != arms[2]['timeline'][341:393]:
        raise ValueError('New-graph Code source states differ')
    if args.check_metadata_only:
        print(json.dumps({'metadata_ok':True,'sources':[source_summary(ep) | {'source_observations':len(ep['timeline'])} for ep in arms]}))
        return
    if args.output_dir is None or args.output_dir.exists():raise ValueError('New --output-dir required')
    os.environ.setdefault('MUJOCO_GL','egl')
    font=choose_font(args.font)
    args.output_dir.mkdir(parents=True)
    comparison_frames,comparison_timeline=render_comparison(arms,args.output_dir,font,args.fps)
    full_frames,full_timeline=render_full(arms[1],args.output_dir,font,args.fps)
    for name,rows_ in (('comparison_timeline.json',comparison_timeline),('full_timeline.json',full_timeline)):
        (args.output_dir/name).write_text(json.dumps(rows_,ensure_ascii=False,indent=2)+'\n')
    files={name:{'sha256':sha(args.output_dir/name),'bytes':(args.output_dir/name).stat().st_size} for name in
           ('comparison_code_aligned.mp4','comparison_cover.png','full_fixed_success.mp4','full_cover.png',
            'comparison_timeline.json','full_timeline.json')}
    manifest={'schema':'p55.sealed-presentation.v1','scope':'Visualization of three existing sealed episodes; no rollout, physics integration, training, or promotion',
              'sources':[source_summary(ep) for ep in arms], 'fps':args.fps,
              'comparison':{'file':'comparison_code_aligned.mp4','source_observations':max(len(ep['timeline'])-341 for ep in arms),
                            'frames_per_observation':4,'video_frames':comparison_frames,'alignment':'Code starts at video frame 0 in all columns; ended columns hold their final recorded image'},
              'full':{'file':'full_fixed_success.mp4','original_observations':len(arms[1]['timeline']),
                      'slow_replay_stages':['s8a','s9'],'replay_frames_per_observation':4,'video_frames':full_frames},
              'files':files,'claim_limit':'Nominal 90% single-cell illustration; online z harms this matched cell and is not an incremental gain.'}
    (args.output_dir/'MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    page='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>P55 封存轨迹视频</title>
<style>body{font:18px system-ui;max-width:1060px;margin:32px auto;background:#101823;color:#e8eef7}a{color:#8dd2ff}video{width:100%;margin:12px 0 30px}small{color:#b7c4d5}</style>
<h1>P55 小书本：旧图／新图固定 z／新图在线 z</h1>
<p>同一名义 90% 书本初态。视频由已封存回合制作，无新增动作或实验。图完成与官方成功分开显示；三臂对比不能证明在线 z 收益。</p>
<h2>Code 起点对齐对比</h2><p>每列外部与腕部视角，动作顺序不变，每个记录帧重复四次；提前结束的列定格。</p>
<video controls preload="metadata" poster="comparison_cover.png" src="comparison_code_aligned.mp4"></video>
<h2>新图固定 z：原初态完整成功回合</h2><p>完整原始动作顺序；片尾 s8a 与 s9 慢放仅重复记录帧。</p>
<video controls preload="metadata" poster="full_cover.png" src="full_fixed_success.mp4"></video>
<p><small><a href="MANIFEST.json">来源与校验</a> · <a href="comparison_timeline.json">对比逐帧来源</a> · <a href="full_timeline.json">完整逐帧来源</a></small></p></html>'''
    (args.output_dir/'index.html').write_text(page)
    print(json.dumps({'output_dir':str(args.output_dir),'comparison_frames':comparison_frames,
                      'full_frames':full_frames,'fps':args.fps,'files':files},ensure_ascii=False))

if __name__=='__main__':main()
