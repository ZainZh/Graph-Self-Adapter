"""Re-encode an existing lossless RGB cache, with zero simulator execution."""
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from prove.fast_adapter.lila_frontend import ThreeFrontends,FRONTENDS
from tools.fast_adapter.p18_run import digest,save_json
R=Path('evaluations/fast_adapter_p21_frontends_v01')
def main(dino):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    p=R/'bank/shard0';assert not (p/'lila.pt').exists()
    rgb=np.load(p/'RGB.npz');enc=ThreeFrontends(dino);values={k:[] for k in FRONTENDS};began=time.time()
    for start in range(0,len(rgb['rgb']),112):
        f=enc.encode_all(rgb['rgb'][start:start+112],rgb['boxes'][start:start+112])
        for k,v in f.items():values[k].append(v.cpu().half())
    for k,v in values.items():torch.save(torch.cat(v).reshape(256,14,*v[0].shape[1:]),p/f'{k}.pt')
    source=json.loads((p/'LOCAL_SOURCE_RECEIPT.json').read_text())
    expected=next(x['sha256'] for x in source['artifacts'] if x['path'].endswith('RGB.npz'))
    assert digest(p/'RGB.npz')==expected
    save_json(p/'RECEIPT.json',{'shard':0,'domains':256,'images':3584,'seconds':time.time()-began,'source':'same lossless RGB produced by local shard0; re-encoded for remote PCA, no simulator rerun','rgb_sha256':expected,'three_frontends_same_rgb':True,'simulator_steps_this_reencoding':0,'artifacts':[{'path':str(f),'sha256':digest(f)} for f in sorted(p.iterdir()) if f.is_file()]})
    print('REENCODE_COMPLETE',flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dino',required=True);main(p.parse_args().dino)
