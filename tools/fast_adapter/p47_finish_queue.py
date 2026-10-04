"""Finite postprocessing after all four P47 workers end; never starts rollouts."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def binding(path):
    path = Path(path)
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='evaluations/fast_adapter_p47_representation_solver_v01')
    parser.add_argument('--deadline-hours', type=float, default=8.)
    args = parser.parse_args()
    root = Path(args.root)
    state = root / 'diagnostics' / 'FINISHER.json'
    state.parent.mkdir(exist_ok=True)
    with state.open('x') as handle:
        handle.write('{}\n')
    batch = root / 'main_batch'
    record = dict(status='waiting_for_complete_batch', pid=os.getpid(), started=time.time(),
                  implementation=binding(__file__), batch=binding(batch / 'FREEZE.json'),
                  automation_scope='wait, aggregate, audit, archive; no rollouts or candidate updates')
    write(state, record)
    try:
        deadline = time.monotonic() + args.deadline_hours * 3600
        while True:
            failed = list(batch.glob('worker_*/FAILED.json'))
            if failed:
                raise RuntimeError('Worker failure; retain partial evidence and require root review: '
                                   + ', '.join(str(path) for path in failed))
            done = list(batch.glob('worker_*/DONE.json'))
            if len(done) == 4:
                break
            if time.monotonic() > deadline:
                raise TimeoutError('Batch not complete by bounded postprocessing deadline')
            time.sleep(20)
        record.update(status='aggregating', workers=[binding(path) for path in sorted(done)])
        write(state, record)
        command = [sys.executable, '-m', 'tools.fast_adapter.p47_aggregate']
        for task in ('cup', 'insert', 'slider'):
            folder = root / 'main' / task
            command += ['--cohort', str(folder / 'EVALUATION_FREEZE.json'), str(folder / 'cells')]
        command += ['--out', str(root / 'summary')]
        subprocess.run(command, check=True)
        summary = json.loads((root / 'summary' / 'RESULTS.json').read_text())
        if summary['status'] != 'complete' or summary['n_completed_cells'] != 1248:
            raise RuntimeError('Complete physical batch did not pass full evidence validation')
        record.update(status='auditing', summary=binding(root / 'summary' / 'RESULTS.json'))
        write(state, record)
        subprocess.run([sys.executable, '-m', 'tools.fast_adapter.p47_decision_audit',
                        '--root', str(root), '--out', str(root / 'audit')], check=True)
        record.update(status='rendering_prespecified_videos', audit=binding(root / 'audit' / 'AUDIT.json'))
        write(state, record)
        media = root / 'presentation' / 'media'
        media.mkdir(parents=True, exist_ok=True)
        videos = []
        titles = {'cup': '杯柄抓持抬升', 'insert': '已持物插入', 'slider': '滑动把手拉开'}
        for task in ('cup', 'insert', 'slider'):
            folder = root / 'main' / task
            frozen = json.loads((folder / 'EVALUATION_FREEZE.json').read_text())
            first = min(cell['condition_seed'] for cell in frozen['cells'])
            cells = {cell['arm']: folder / 'cells' / f'cell_{index:06d}'
                     for index, cell in enumerate(frozen['cells'])
                     if cell['condition_seed'] == first and cell['rep'] == 'relative_fusion'
                     and cell['model_seed'] == 11}
            for left, right, label in (('fixed', 'continuous', '固定与梯度更新'),
                                       ('continuous', 'direct', '梯度更新与直接求解')):
                output = media / (task + '_' + left + '_' + right)
                reason = f'结果前预定的首个OOD条件{first}，R表征、模型11；无论成功或失败均保留此例。'
                subprocess.run([sys.executable, '-m', 'tools.fast_adapter.p47_video', 'pair',
                                '--left', str(cells[left]), '--right', str(cells[right]),
                                '--out', str(output), '--view', 'detail',
                                '--selection-reason', reason], check=True)
                videos.append(dict(task=task, title=titles[task] + '：' + label,
                                   path=str(output / 'annotated.mp4'), manifest=str(output / 'MANIFEST.json'),
                                   kind='预定首例', condition_seed=first,
                                   video_binding=binding(output / 'annotated.mp4'),
                                   manifest_binding=binding(output / 'MANIFEST.json')))
        write(media / 'VIDEO_INDEX.json', dict(schema='prove.p47-video-index.v1', videos=videos,
                                               results=binding(root / 'summary' / 'RESULTS.json')))
        record.update(status='complete', audit=binding(root / 'audit' / 'AUDIT.json'),
                      completed=time.time())
        write(state, record)
        # Include all sealed RGB archives (only prescribed video cells produce
        # them) as well as every trace and source snapshot. No silent omissions.
        archive = Path('/tmp/prove_p47_complete_records_20260927.tar.gz')
        if archive.exists():
            raise FileExistsError('Preserve previous archive: ' + str(archive))
        with tarfile.open(archive, 'w:gz', compresslevel=1) as handle:
            handle.add(root, arcname=str(root), recursive=True)
        receipt = dict(status='ready', archive=binding(archive), bytes=archive.stat().st_size,
                       all_campaign_files_included=True, created=time.time())
        write(root / 'diagnostics' / 'TRANSFER_READY.json', receipt)
        print(json.dumps(receipt), flush=True)
    except Exception as error:
        record.update(status='failed', error_type=type(error).__name__, error=str(error), ended=time.time())
        write(state, record)
        raise


if __name__ == '__main__':
    main()
