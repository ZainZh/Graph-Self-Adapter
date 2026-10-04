"""Resume after an input-path repair; unchanged frozen v03 training code."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
from tools.fast_adapter.p40_launch_v03 import ROOT,run

def main():
    source=ROOT/'insert/data_v02/TRAIN.pt';alias=ROOT/'insert/data/TRAIN.pt'
    if not source.exists():raise RuntimeError('Insertion bank missing')
    if not alias.exists():alias.symlink_to('../data_v02/TRAIN.pt')
    assert alias.resolve()==source.resolve()
    failed=ROOT/'insert/effects_v03'
    if failed.exists():
        assert not list(failed.iterdir()),'Only the empty failed output directory may be moved'
        failed.rename(ROOT/'insert/effects_v03_failed_path')
    record=dict(time=time.time(),pid=os.getpid(),input_path_repair=str(alias),target=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),note='No model/algorithm/data changed; previous missing-file failure retained; existing cup/hammer heads reused')
    (ROOT/'INPUT_PATH_REPAIR.json').write_text(json.dumps(record,indent=2))
    run(['effects','--task','insert'],2,'effects_insert_path_repair')
    jobs=[]
    for gpu in range(4):
        with (ROOT/'logs'/f'resume_worker{gpu}.log').open('x') as f:
            jobs.append(subprocess.Popen([sys.executable,'-u','-m','tools.fast_adapter.p40_launch_v03','--worker',str(gpu)],stdout=f,stderr=subprocess.STDOUT))
    exits=[p.wait() for p in jobs]
    (ROOT/'TRAIN_v03_RESUMED_DONE.json').write_text(json.dumps(dict(time=time.time(),exits=exits)))
    if any(exits):raise RuntimeError(exits)

if __name__=='__main__':main()
