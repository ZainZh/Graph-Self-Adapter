"""Bounded postprocessing for existing root-started P43 validation jobs.

This worker never starts trials, changes candidates, or reads partial outcomes.
"""
import argparse,json,os,subprocess,sys,time
from pathlib import Path


def running(pid,name):
    proc=Path('/proc')/str(pid)
    if not proc.exists():return False
    try:
        state=(proc/'stat').read_text().split()[2]
        command=(proc/'cmdline').read_bytes().decode(errors='replace')
    except FileNotFoundError:return False
    return state!='Z' and 'p43_validate_frozen' in command and name in command


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--root',default='evaluations/fast_adapter_p43_neural_geometry_v01');parser.add_argument('--max-wait-seconds',type=int,default=5400);args=parser.parse_args()
    root=Path(args.root);launch=json.loads((root/'independent_freeze/LAUNCH.json').read_text());deadline=time.monotonic()+args.max_wait_seconds
    complete=set();summary=root/'training/INDEPENDENT_FINALIZATION.json'
    if summary.exists():raise FileExistsError('Preserve the existing finalization receipt')
    def receipt(status,**fields):
        summary.parent.mkdir(parents=True,exist_ok=True)
        payload=dict(status=status,time=time.time(),audited_models=sorted(complete),**fields)
        temporary=summary.with_suffix('.tmp');temporary.write_text(json.dumps(payload,indent=2)+'\n');temporary.replace(summary)
    receipt('waiting_for_closed_batches')
    try:
        while len(complete)<4:
            progress={}
            for job in launch:
                name=job['name']
                if name in complete:continue
                freeze=root/'independent_freeze'/f'{name}.json';frozen=json.loads(freeze.read_text());out=Path(frozen['execution']['out'])
                expected=[out/f'{split}_{seed}_{arm}' for seed,split in zip(frozen['seeds'],frozen['splits']) for arm in frozen['arms']]
                count=sum((p/'RESULT.json').exists() for p in expected);alive=running(job['pid'],name);progress[name]=dict(complete_cells=count,wrapper_running=alive)
                if alive:continue
                if count!=96 or len(list(out.glob('*/RESULT.json')))!=96:raise RuntimeError(f'{name} wrapper ended with incomplete or extra cells: {count}')
                if not all(all((p/file).exists() for file in ('SEAL.json','TRACE.npy','OBS.npy','TERMINAL.npy')) for p in expected):raise RuntimeError(f'{name} is missing sealed artifacts')
                log=root/'training'/f'{name}_independent_audit.log'
                with log.open('w') as stream:
                    subprocess.run([sys.executable,'-m','tools.fast_adapter.p43_core_independent_audit','--freeze',str(freeze)],check=True,stdout=stream,stderr=subprocess.STDOUT)
                complete.add(name);print(json.dumps(dict(audit_complete=name)),flush=True)
            receipt('waiting_for_closed_batches' if len(complete)<4 else 'all_models_audited',progress=progress)
            if len(complete)==4:break
            if time.monotonic()>deadline:raise TimeoutError('Bounded postprocessing wait expired; trials were not interrupted')
            time.sleep(20)
        with (root/'training/independent_report.log').open('w') as stream:
            subprocess.run([sys.executable,'-m','tools.fast_adapter.p43_core_independent_report','--root',str(root)],check=True,stdout=stream,stderr=subprocess.STDOUT)
        receipt('complete',cells=384,report=str(root/'training/INDEPENDENT_SUMMARY.json'));print('P43 independent postprocessing complete',flush=True)
    except Exception as error:
        receipt('failed',error=repr(error));raise

if __name__=='__main__':main()
