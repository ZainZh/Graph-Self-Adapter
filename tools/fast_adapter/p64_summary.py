#!/usr/bin/env python3
"""Combine completed P64 task audits; never read partial rollout directories."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def reference(path):
    path=Path(path).resolve()
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):digest.update(block)
    return dict(path=str(path),sha256=digest.hexdigest())


def require(value,message):
    if not value:raise ValueError(message)


def cup_rows(source):
    require(source.get('schema')=='prove.p64-cup-complete-public-runtime-audit.v1'
            and source.get('all_runtime_invariants_passed') is True
            and len(source.get('cells',[]))==24 and len(source.get('pairs',[]))==12,
            'Completed 24-cell cup audit required')
    rows=[]
    for group in ('id','size','position'):
        pairs=[p for p in source['pairs'] if p['split']==group]
        require(len(pairs)==4 and len({p['seed'] for p in pairs})==4,'Cup pairing denominator differs')
        rescues=sum(bool(p['online_success']) and not bool(p['fixed_success']) for p in pairs)
        harms=sum(bool(p['fixed_success']) and not bool(p['online_success']) for p in pairs)
        require(rescues==sum(p['rescue'] for p in pairs) and harms==sum(p['harm'] for p in pairs),
                'Cup rescue/harm flags differ from outcomes')
        arms={}
        for display,original,success_key in (('fixed','fixed','fixed_success'),
                                              ('continuous','translation_size5','online_success')):
            value=source['groups'][group+'/'+original]
            require(value['n']==4 and value['success']==sum(bool(p[success_key]) for p in pairs),
                    'Cup group and pair success counts differ')
            require(0<=value['native_sustained_success']<=4 and 0<=value['mechanical_anomaly']<=4,
                    'Invalid cup count')
            arms[display]=dict(success=value['success'],denominator=4,
                               native_success=value['native_sustained_success'],
                               mechanical_anomalies=value['mechanical_anomaly'])
        rows.append(dict(stratum=group,arms=arms,paired_denominator=4,rescues=rescues,harms=harms))
    return rows


def book_rows(source):
    require(source.get('schema')=='prove.p64-book-audit.v1' and source.get('complete') is True
            and len(source.get('records',[]))==24,'Completed 24-cell book audit required')
    require(all(source.get('audit',{}).values()) and source.get('audit'), 'Book runtime audit incomplete')
    rows=[]
    for group in ('id','size','position_shift'):
        value=source['groups'][group]
        pairs=value['pairs']
        require(value['paired_denominator']==4 and len(pairs)==4
                and len({p['condition'] for p in pairs})==4,'Book pair denominator differs')
        require(value['rescues']==sum(p['rescue'] for p in pairs)
                and value['harms']==sum(p['harm'] for p in pairs),'Book pair counts differ')
        arms={}
        for display,original in (('fixed','fixed'),('continuous','online')):
            v=value['arms'][original]
            records=[r for r in source['records'] if r['arm']==original and r['condition']['group']==group]
            require(len(records)==4 and v['full_denominator']==4
                    and len({r['condition']['id'] for r in records})==4,'Book full denominator differs')
            entered=sum(bool(r['entered']) for r in records)
            actions=sum(r['steps'] for r in records)
            require(v['entered_denominator']==entered and v['executed_actions']==actions
                    and v['strict_success']==sum(bool(r['strict']) for r in records)
                    and v['native_success']==sum(bool(r['native']) for r in records),
                    'Book aggregate differs from per-cell evidence')
            require(all(0<=r['steps']<=24 for r in records),'Book H24 budget exceeded')
            arms[display]=dict(success=v['strict_success'],denominator=4,
                               native_success=v['native_success'],entered_denominator=entered,
                               entry_failures=4-entered,original_entry_failures=v['original_entry_failures'],
                               executed_actions=actions,mean_actions_all_conditions=actions/4,
                               mean_actions_entered=actions/entered if entered else None)
        rows.append(dict(stratum=group,arms=arms,paired_denominator=4,
                         rescues=value['rescues'],harms=value['harms']))
    return rows


def build(cup_summary,book_audit,output,*,cup_video_dir=None,book_video_dir=None):
    cup_summary,book_audit,output=map(Path,(cup_summary,book_audit,output))
    cup,book=read(cup_summary),read(book_audit)
    tasks=dict(cup=dict(strata=cup_rows(cup),success_definition='validated cup grasp/lift success',
                        scope=cup['scope']),
               book=dict(strata=book_rows(book),success_definition='native goal and book contact jointly hold3',
                         scope=book['interpretation']['scope'],
                         position_scope=book['interpretation']['position']))
    cup_video_dir=Path(cup_video_dir) if cup_video_dir else cup_summary.parent
    book_video_dir=Path(book_video_dir) if book_video_dir else book_audit.parent.parent/'summary_v01'
    selected=(('cup_id',cup_video_dir/'id_640100_paired.mp4'),
              ('cup_size',cup_video_dir/'size_640100_paired.mp4'),
              ('book_id',book_video_dir/'id_source62/comparison.mp4'),
              ('book_size',book_video_dir/'size_source66/comparison.mp4'))
    videos={name:reference(path) for name,path in selected}
    value=dict(schema='prove.p64-combined-summary.v1',complete=True,
               sources=dict(cup_summary=reference(cup_summary),book_audit=reference(book_audit)),
               method='Shared LiLa/DINO plus public-contour persistent-z design; task-specific training and checkpoints, not zero-shot transfer',
               conditions='Separate ID, size-only and position-only strata; no orientation OOD; no pooled cross-task rate',
               time_metric='Book mean executed physical actions, not wall-clock duration; full and entered denominators both shown',
               tasks=tasks,videos=videos)
    require(not (output/'ROOT_SUMMARY.json').exists() and not (output/'index.html').exists(),
            'Refusing to overwrite a summary')
    output.mkdir(parents=True,exist_ok=True)
    page=['<!doctype html><html lang="zh"><meta charset="utf-8"><title>P64 跨任务验证</title>',
          '<style>body{font:16px system-ui;max-width:1200px;margin:36px auto;padding:0 20px;line-height:1.6;color:#192532}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{border:1px solid #ddd;padding:8px;text-align:left}a{color:#1459aa}small{color:#536273}</style>',
          '<h1>P64：位置与尺寸的跨任务验证</h1>',
          '<p>共享 LiLa／DINO、公开轮廓与持续 z 更新设计。两个任务分别训练，使用各自 checkpoint；不是零样本迁移。无朝向 OOD，不合并任务分母。</p>']
    titles={'cup':'杯柄抓持／提起','book':'真实 LIBERO 近入口插书（s9，H24）'}
    labels={'id':'ID','size':'仅尺寸变化','position':'仅位置变化','position_shift':'收纳盒 ±4 mm 入口位置变化'}
    for task,data in tasks.items():
        page+=['<h2>'+titles[task]+'</h2>',
               '<table><tr><th>条件组</th><th>模式</th><th>严格成功／全组</th><th>native／全组</th>'+
               ('<th>进入分母</th><th>平均动作步／进入</th><th>平均动作步／全组</th>' if task=='book' else '<th>机械异常</th>')+
               '<th>配对救回／误伤</th></tr>']
        for row in data['strata']:
            for arm,v in row['arms'].items():
                extra=(f'<td>{v["entered_denominator"]}/4</td><td>{v["mean_actions_entered"]:.2f}</td>'
                       f'<td>{v["mean_actions_all_conditions"]:.2f}</td>' if task=='book' and v['entered_denominator'] else
                       f'<td>0/4</td><td>不适用</td><td>{v["mean_actions_all_conditions"]:.2f}</td>' if task=='book' else
                       f'<td>{v["mechanical_anomalies"]}</td>')
                page.append(f'<tr><td>{labels[row["stratum"]]}</td><td>{"固定 z" if arm=="fixed" else "持续 z"}</td>'
                            f'<td>{v["success"]}/{v["denominator"]}</td><td>{v["native_success"]}/{v["denominator"]}</td>'+extra+
                            f'<td>{row["rescues"]}/{row["harms"]}</td></tr>')
        page.append('</table>')
        if task=='book':
            page.append('<p>旧入口失败保留在四条件分母。动作步只计物理控制步，不是训练、推理或总墙钟时间；进入分母均值包含进入后的全部成功与失败。局部插书不代表完整任务图完成。</p>')
            page.append('<p>'+html.escape(data['position_scope'])+'</p>')
        page.append('<p>预选配对视频： '+' · '.join(
            '<a href="'+html.escape(os.path.relpath(videos[task+'_'+group]['path'],output),quote=True)+'">'+label+'</a>'
            for group,label in (('id','ID'),('size','尺寸变化')) )+'</p>')
    page.append('<small>来源文件和四个视频的 SHA-256 见 <a href="ROOT_SUMMARY.json">ROOT_SUMMARY.json</a>。</small></html>')
    (output/'ROOT_SUMMARY.json').write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    (output/'index.html').write_text('\n'.join(page)+'\n')
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cup-summary',type=Path,required=True)
    parser.add_argument('--book-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cup-video-dir',type=Path)
    parser.add_argument('--book-video-dir',type=Path)
    args=parser.parse_args()
    build(args.cup_summary,args.book_audit,args.output,
          cup_video_dir=args.cup_video_dir,book_video_dir=args.book_video_dir)


if __name__=='__main__':main()
