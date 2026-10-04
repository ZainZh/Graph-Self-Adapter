"""Read-only P69 batch accounting; missing cells stay in the denominator."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rows(path):return [json.loads(line) for line in path.read_text().splitlines() if line.strip()] if path.exists() else []


def stage(directory,prefix,code,result):
    if result.get('graph_complete'):return None
    if code.get('qualified'):return 's9'
    path=directory/('code/TRACE.jsonl' if prefix.get('qualified') else 'prefix/TRACE.jsonl')
    trace=rows(path)
    if trace:
        last=trace[-1]
        if last.get('stage_id'):return last['stage_id']
        for event in reversed(last.get('events',[])):
            if event.get('judge',{}).get('stage_id'):return event['judge']['stage_id']
    return 'code_unresolved' if prefix.get('qualified') else 'prefix_unresolved'


TASK_STAGES=('s1','s2','s3','s4','s5','s6','s7','s8','s8a','s9')


def stage_progress(directory,result,seal):
    """Count explicit sealed stage observations and completion evidence, not order inference."""
    reached=set();passed=set()
    def sealed_rows(name):
        path=directory/name
        if not path.exists():return []
        if seal['files'].get(name)!=sha(path):raise ValueError('Stage evidence differs from seal: '+name)
        return read(path) if name.endswith('.json') else rows(path)
    for row in sealed_rows('prefix/TRACE.jsonl'):
        if row.get('phase')=='settling' or row.get('stage_id')=='settling':continue
        for event in row.get('events',[]):
            judge=event.get('judge',{})
            sid=judge.get('stage_id')
            if sid not in TASK_STAGES:continue
            reached.add(sid)
            if judge.get('state')=='complete':passed.add(sid)
    for row in sealed_rows('code/TRACE.jsonl'):
        sid=row.get('stage_id')
        if sid in TASK_STAGES:reached.add(sid)
        for transition in row.get('transitions',[]):
            previous=transition.get('from_stage')
            if previous in TASK_STAGES:reached.add(previous);passed.add(previous)
    # Root-level handoffs verify completion, but never invent unobserved destinations.
    for transition in sealed_rows('STAGE_TRANSITIONS.json'):
        previous=transition.get('from_stage')
        if previous in reached:passed.add(previous)
    execution=sealed_rows('public/EXECUTED.jsonl')
    if execution:
        reached.add('s9')
        if result.get('graph_complete') and result.get('official_success'):passed.add('s9')
    return dict(reached=[s for s in TASK_STAGES if s in reached],passed=[s for s in TASK_STAGES if s in passed])


def cell_summary(cell,directory,freeze_path):
    row=dict(stage_progress=dict(reached=[],passed=[]),cell=cell,status='not_started',full_success=False,entry=False,failure_stage=None,total_actions=None,updates=None)
    if directory is None:return row
    row['path']=str(directory)
    row['bindings']={str(p.relative_to(directory)):sha(p) for p in directory.rglob('*') if p.is_file() and
        p.name in ('RESULT.json','PREFIX_RESULT.json','CODE_RESULT.json','ABORT.json','TRACE.jsonl','STAGE_TRANSITIONS.json','CELL.json','SEAL.json')}
    prefix=read(directory/'PREFIX_RESULT.json') if (directory/'PREFIX_RESULT.json').exists() else {}
    code=read(directory/'CODE_RESULT.json') if (directory/'CODE_RESULT.json').exists() else {}
    result=read(directory/'RESULT.json') if (directory/'RESULT.json').exists() else {}
    row.update(prefix=prefix,code=code,transitions=read(directory/'STAGE_TRANSITIONS.json') if (directory/'STAGE_TRANSITIONS.json').exists() else [],
               entry=bool(code.get('qualified')),failure_stage=stage(directory,prefix,code,result))
    if (directory/'ABORT.json').exists():
        row.update(status='error',abort=read(directory/'ABORT.json'));return row
    if not result or not (directory/'SEAL.json').exists():row['status']='incomplete';return row
    seal=read(directory/'SEAL.json')
    if result.get('cell')!=cell or result.get('seal_sha256')!=sha(directory/'SEAL.json'):
        raise ValueError('Result identity/seal mismatch: '+str(directory))
    if result.get('schema')!='prove.p69-full-rsi-result.v1' or result.get('full_task_success_evidence') is not True:
        row['status']='invalid_non_task_evidence';return row
    if seal.get('binding_sha256')!=sha(freeze_path):raise ValueError('Seal freeze mismatch')
    row['stage_progress']=stage_progress(directory,result,seal)
    row.update(status='complete',full_success=bool(result.get('graph_complete') and result.get('official_success')),
        total_actions=result.get('total_actions'),updates=result.get('updates'),stop_reason=result.get('stop_reason'),
        entry=bool(result.get('near_entry_qualified')))
    if not isinstance(row['total_actions'],int) or not 0<=row['total_actions']<=544:raise ValueError('Invalid full action budget')
    return row


def summarize(manifest):
    split=manifest['split']
    if split not in ('explore','test'):raise ValueError('Only separate explore or test task batches; no training fits')
    versions=[];identity=None;seen_versions=set()
    for item in manifest['versions']:
        freeze_path=Path(item['freeze']);freeze=read(freeze_path);cells=freeze['cells'];version=item['version']
        if version in seen_versions:raise ValueError('Duplicate version')
        seen_versions.add(version)
        if freeze.get('split')!=split or freeze.get('version')!=version or len(cells)!=10:raise ValueError('Wrong split/version/denominator')
        current={(str(c['init_id']),c.get('state_sha256')) for c in cells}
        if len(current)!=10 or any(c['split']!=split or c['version']!=version for c in cells):raise ValueError('Cell identity mismatch')
        if identity is not None and current!=identity:raise ValueError('Versions must pair identical initial states')
        identity=current;root=Path(item['cells_root']);directories={}
        for marker in root.glob('*/CELL.json'):
            c=read(marker)
            if c not in cells or c['index'] in directories:raise ValueError('Unscheduled or duplicate cell')
            directories[c['index']]=marker.parent
        # An interrupted mkdir before CELL is written remains incomplete, never success.
        for directory in root.glob('cell_*'):
            if directory.is_dir() and not (directory/'CELL.json').exists():
                try:index=int(directory.name.removeprefix('cell_'))
                except ValueError:raise ValueError('Unidentified partial cell directory')
                if index not in {c['index'] for c in cells}:raise ValueError('Unscheduled partial cell')
                directories[index]=directory
        records=[cell_summary(c,directories.get(c['index']),freeze_path) for c in cells]
        omitted={'s6'} if freeze.get('graph',{}).get('prefix_handoff_after')=='s5' else set()
        for record in records:
            record['stage_progress']['not_applicable']=sorted(omitted)
            if omitted.intersection(record['stage_progress']['reached']+record['stage_progress']['passed']):
                raise ValueError('Observed stage contradicts frozen omitted node')
        versions.append(dict(version=version,freeze=dict(path=str(freeze_path),sha256=sha(freeze_path)),
            n=10,full_success=sum(r['full_success'] for r in records),entered=sum(r['entry'] for r in records),
            stages={sid:(dict(status='not_applicable',reason='omitted_by_frozen_graph',reached=None,passed=None,n=None) if sid in omitted else dict(status='applicable',reached=sum(sid in r['stage_progress']['reached'] for r in records),passed=sum(sid in r['stage_progress']['passed'] for r in records),n=10)) for sid in TASK_STAGES},
            status_counts=dict(Counter(r['status'] for r in records)),
            failure_stage_counts=dict(Counter(r['failure_stage'] for r in records if r['failure_stage'])),
            total_actions=sum(r['total_actions'] or 0 for r in records),updates=sum(r['updates'] or 0 for r in records),cells=records))
    paired=[dict(init_id=init_id,state_sha256=digest,versions={v['version']:next({k:r[k] for k in ('status','full_success','entry','failure_stage','total_actions','updates','stage_progress')}
        for r in v['cells'] if str(r['cell']['init_id'])==init_id) for v in versions}) for init_id,digest in sorted(identity or [])]
    return dict(schema='prove.p69-batch-summary.v1',split=split,versions=versions,paired_initial_states=paired,
        all_complete=all(v['status_counts'].get('complete')==10 for v in versions) and bool(versions),
        stage_policy='Reached requires a sealed judge/Code observation or actual s9 record; passed requires explicit completion/transition. Settling is not a task stage; unstarted stages never count as passed.',
        sampling_limit='Only ten paired initial states per version; descriptive engineering evidence, not stable-generalization confirmation. Explore and test remain separate reports.',
        denominator_policy='Ten predeclared initial states per version including missing/error; partial counts are not final task rates.',
        audit_scope='Read-only result identity and seal reference verification; not a full sealed-file audit.',physics_steps=0)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();value=summarize(read(args.manifest));value['manifest']=dict(path=str(args.manifest),sha256=sha(args.manifest))
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'SUMMARY.json').write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    text=['# P69 '+value['split']+' 批量汇总','', '固定每版10个初态；未启动／不完整／程序错误保留分母。训练拟合不计任务成功。未完整批次仅为进度，不是最终成功率。','',
          '| 版本 | 完整成功/10 | 进入s9/10 | 状态计数 | 已完成动作总数 | 更新总数 |','|---|---:|---:|---|---:|---:|']
    for v in value['versions']:text.append(f"| {v['version']} | {v['full_success']}/10 | {v['entered']}/10 | {v['status_counts']} | {v['total_actions']} | {v['updates']} |")
    text+=['','| 版本 | 阶段 | 达到/10 | 通过/10 |','|---|---|---:|---:|']
    for v in value['versions']:
        for sid,counts in v['stages'].items():
            if counts['status']=='not_applicable':text.append(f"| {v['version']} | {sid} | 不适用（冻结图省略） | 不计通过或失败 |")
            else:text.append(f"| {v['version']} | {sid} | {counts['reached']}/10 | {counts['passed']}/10 |")
    text+=['','Settling不计任务阶段；只计封存日志明确达到／通过的阶段。explore和test严格分开；每版10初态仅为快速工程证据，不能宣称稳定泛化。','','逐初态配对、失败阶段及来源hash见 [SUMMARY.json](SUMMARY.json)。只核验结果身份与seal引用，不替代完整轨迹审计。']
    (args.output/'SUMMARY.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
