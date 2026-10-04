"""Inventory and archive a closed P40 directory once; no training mutation."""
import hashlib
import json
import tarfile
import time
from pathlib import Path

ROOT=Path('evaluations/fast_adapter_p40_skills_v01')


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    assert (ROOT/'POSTPROCESS_DONE.json').exists()
    freeze=json.loads((ROOT/'FINAL_FREEZE.json').read_text())
    bindings=freeze['sources']+freeze['models']+freeze['visual_weights']
    for b in bindings:assert digest(Path(b['path']))==b['sha256'],b['path']
    archive=Path('/tmp/prove_p40_closed_v01.tar.gz')
    assert not archive.exists()
    files=[];retained=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file():continue
        if p.name.startswith('EXPORT_'):continue
        # Full original frame features and pickled scene collection records stay
        # on the server with hashes. Compressed observed training banks travel.
        remote_only=(p.name=='LOCAL_FEATURES.pt' or p.suffix=='.npy')
        item=dict(path=str(p),sha256=digest(p),bytes=p.stat().st_size,
                  location='server retained original' if remote_only else 'exported')
        (retained if remote_only else files).append(item)
    manifest=ROOT/'EXPORT_MANIFEST.json'
    with manifest.open('x') as f:json.dump(dict(time=time.time(),server='/home/ubuntu/github/prove-p40-skills',files=files,retained=retained,frozen_bindings_verified=len(bindings)),f,indent=2)
    with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
        for row in files:tf.add(row['path'],arcname=row['path'],recursive=False)
        tf.add(manifest,arcname=str(manifest))
    receipt=dict(time=time.time(),archive=str(archive),sha256=digest(archive),bytes=archive.stat().st_size,files=len(files),retained_files=len(retained))
    with (ROOT/'EXPORT_RECEIPT.json').open('x') as f:json.dump(receipt,f,indent=2)
    print(receipt)


if __name__=='__main__':main()
