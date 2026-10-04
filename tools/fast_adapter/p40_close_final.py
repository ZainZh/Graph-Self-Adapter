"""Finite postprocessing, after every frozen independent execution has sealed."""
import json
import subprocess
import sys
import time
from tools.fast_adapter.p40_launch_v03 import ROOT, environment


def main():
    deadline = time.time()+10800
    while not (ROOT/'FINAL_DONE.json').exists():
        if time.time()>deadline:
            raise RuntimeError('Final dependency timeout; retain outputs and inspect queue')
        time.sleep(10)
    done=json.loads((ROOT/'FINAL_DONE.json').read_text())
    assert done['exits']==[0,0,0], done
    for task in ('cup','hammer','insert'):
        assert (ROOT/task/'FINAL_DONE.json').exists()
    with (ROOT/'logs/FINAL_REPORT.log').open('x') as f:
        subprocess.run([sys.executable,'-u','-m','tools.fast_adapter.p40_report_v02','--phase','final'],env=environment(0),stdout=f,stderr=subprocess.STDOUT,check=True)
    jobs=[]
    for gpu,task in enumerate(('cup','hammer','insert')):
        with (ROOT/'logs'/f'FINAL_AUDIT_{task}.log').open('x') as f:
            jobs.append(subprocess.Popen([sys.executable,'-u','-m','tools.fast_adapter.p40_audit_v02','--phase','final','--task',task],env=environment(gpu),stdout=f,stderr=subprocess.STDOUT))
    exits=[p.wait() for p in jobs]
    assert exits==[0,0,0],exits
    for module,tag in (('tools.fast_adapter.p40_diagnostics','DIAGNOSTICS'),('tools.fast_adapter.p40_media','MEDIA'),('tools.fast_adapter.p40_present','PRESENT')):
        with (ROOT/'logs'/('FINAL_'+tag+'.log')).open('x') as f:
            subprocess.run([sys.executable,'-u','-m',module],env=environment(0),stdout=f,stderr=subprocess.STDOUT,check=True)
    with (ROOT/'POSTPROCESS_DONE.json').open('x') as f:
        json.dump(dict(time=time.time(),status='complete; awaiting root scientific review'),f)


if __name__=='__main__':
    main()
