"""Local integrity, independent tabulation and freshness checks after download."""
import collections
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time

ROOT=Path('evaluations/fast_adapter_p42_geometry_scale_v01')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def geometry_digest(cfg):
    value={k:v for k,v in cfg.items() if k not in ['seed','split']}
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()


def main():
    out=ROOT/'summary';manifest=json.loads((out/'REMOTE_MANIFEST.json').read_text())
    mismatches=[]
    for item in manifest['files']:
        path=ROOT/item['path']
        if not path.is_file() or path.stat().st_size!=item['bytes'] or sha(path)!=item['sha256']:
            mismatches.append(item['path'])
    assert not mismatches,mismatches
    for path,digest in manifest['executed_tool_sources'].items():
        assert sha(path)==digest,path
    write(out/'TRANSFER_VERIFICATION.json',dict(status='all_remote_artifact_bytes_verified_locally',
        remote=manifest['remote'],verified_at=time.time(),files=len(manifest['files']),
        bytes=sum(i['bytes'] for i in manifest['files']),mismatches=mismatches,
        remote_manifest_sha256=sha(out/'REMOTE_MANIFEST.json')))
    result=json.loads((out/'RESULTS.json').read_text());rows=result['rows']
    protocol=json.loads((ROOT/'PROTOCOL.json').read_text())
    freeze=json.loads((ROOT/'FREEZE.json').read_text())
    frozen_time=datetime.fromisoformat(freeze['frozen_at_utc']).timestamp()
    expected={(t,s,n,a) for t in protocol['tasks'] for s in protocol['splits'] for n in protocol['seeds'] for a in protocol['arms']}
    assert {(r['task'],r['split'],r['seed'],r['arm']) for r in rows}==expected
    assert len(rows)==1280 and not result['exceptions']
    grouped=collections.defaultdict(dict)
    for row in rows:
        raw=json.loads(Path(row['source']).read_text())
        assert raw['success']==row['success']
        pre=json.loads((Path(row['source']).parent/'PRE_RUN.json').read_text())
        contract=json.loads((Path(row['source']).parent/'P42_EPISODE_CONTRACT.json').read_text())
        assert frozen_time < contract['start_time'] <= pre['time']
        grouped[row['task'],row['split'],row['seed']][row['arm']]=row
    for cell in result['cells']:
        selected=[r for r in rows if (r['task'],r['split'],r['arm'])==(cell['task'],cell['split'],cell['arm'])]
        k=sum(r['success'] for r in selected);n=len(selected);z=1.959963984540054;p=k/n
        assert n==64 and k==cell['successes']
        center=(p+z*z/(2*n))/(1+z*z/n)
        half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        assert max(abs(a-b) for a,b in zip(cell['wilson_95'],[max(0.,center-half),min(1.,center+half)]))<1e-12
    for pair in result['paired']:
        d=[int(grouped[pair['task'],pair['split'],seed]['gradient']['success'])-
           int(grouped[pair['task'],pair['split'],seed][pair['reference']]['success']) for seed in protocol['seeds']]
        w=d.count(1);l=d.count(-1)
        assert (w,l,d.count(0))==(pair['wins'],pair['losses'],pair['ties'])
        assert abs(sum(d)/len(d)-pair['delta'])<1e-15
        p=sum(math.comb(w+l,i) for i in range(w,w+l+1))/(2**(w+l)) if w+l else 1.
        assert abs(p-pair['exact_mcnemar_one_sided_p'])<1e-14
    family=sorted([p for p in result['paired'] if p['primary']],key=lambda p:p['exact_mcnemar_one_sided_p'])
    value=0.
    assert len(family)==4
    for i,p in enumerate(family):
        value=max(value,(4-i)*p['exact_mcnemar_one_sided_p'])
        assert abs(min(1.,value)-p['holm_adjusted_p'])<1e-14
    old_digests=set()
    for path in Path('evaluations/fast_adapter_p41_geometry3d_v01').rglob('PRE_RUN.json'):
        pre=json.loads(path.read_text())
        if 'cfg' in pre:old_digests.add(geometry_digest(pre['cfg']))
    fresh={}
    for key,arms in grouped.items():
        digests=[]
        for row in arms.values():
            pre=json.loads((Path(row['source']).parent/'PRE_RUN.json').read_text())
            digests.append(geometry_digest(pre['cfg']))
        assert len(set(digests))==1,key
        fresh['/'.join(map(str,key))]=digests[0]
    assert len(fresh)==256 and len(set(fresh.values()))==256
    assert not (set(fresh.values())&old_digests)
    write(out/'FRESHNESS.json',dict(unique_new_conditions=256,paired_arms_identical_geometry=True,
        prior_p41_unique_configurations=len(old_digests),overlap_with_prior_p41=[],
        fingerprint='canonical complete cfg excluding seed/split; includes actual physical geometry',
        conditions=fresh))
    videos=json.loads((out/'VIDEOS.json').read_text());video_checks=[]
    for video in videos:
        path=out/video['file'];assert sha(path)==video['sha256']
        value=json.loads(subprocess.check_output(['ffprobe','-v','error','-show_streams','-show_format','-of','json',str(path)],text=True))
        streams=[s for s in value['streams'] if s['codec_type']=='video']
        assert len(streams)==1 and streams[0]['width']==1344 and streams[0]['height']==1120
        assert int(streams[0]['nb_frames'])==220
        assert abs(float(value['format']['duration'])-11.)<.01
        video_checks.append(dict(file=video['file'],frames=220,duration_seconds=11.,sha256=sha(path)))
    write(out/'LOCAL_VERIFICATION.json',dict(status='passed',verified_at=time.time(),
        all_1280_raw_results_match_summary=True,all_20_success_cells_and_wilson_intervals_recomputed=True,
        all_episodes_started_after_protocol_freeze=True,
        local_and_remote_executed_tool_sources_identical=True,
        all_16_paired_win_loss_tie_and_exact_p_recomputed=True,four_primary_holm_values_recomputed=True,
        paired_bootstrap='generated by preregistered frozen server script; source and output hashes verified; not recomputed locally',
        fresh_conditions_verified=True,videos=video_checks,
        source_sha256=sha('tools/fast_adapter/p42_verify_local.py')))
    print(json.dumps(dict(files=len(manifest['files']),bytes=sum(i['bytes'] for i in manifest['files']),conditions=256,videos=len(videos),status='passed')))


if __name__=='__main__':main()
