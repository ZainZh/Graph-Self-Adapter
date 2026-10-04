"""Verify the closed P40 archive before importing; preserve existing files."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path

ROOT=Path('evaluations/fast_adapter_p40_skills_v01')


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('receipt',type=Path);a=p.parse_args()
    receipt=json.loads(a.receipt.read_text())
    assert digest(a.archive)==receipt['sha256']
    with tarfile.open(a.archive,'r:gz') as tf:
        manifest=json.load(tf.extractfile(str(ROOT/'EXPORT_MANIFEST.json')))
        for row in manifest['files']:
            path=Path(row['path'])
            assert path.resolve().is_relative_to(ROOT.resolve())
            if path.exists():assert digest(path)==row['sha256'],f'Preserve conflicting local file: {path}'
        for member in tf.getmembers():
            dest=Path(member.name)
            assert dest.resolve().is_relative_to(ROOT.resolve()),member.name
            assert member.isfile() or member.issym() or member.isdir(),member.name
            if member.issym():assert (dest.parent/member.linkname).resolve().is_relative_to(ROOT.resolve())
        tf.extractall('.')
    for row in manifest['files']:
        assert digest(Path(row['path']))==row['sha256'],row['path']
    result=dict(archive_sha256=receipt['sha256'],files_verified=len(manifest['files']),retained_server_files=len(manifest['retained']),status='closed export verified without overwriting differing local artifacts')
    with (ROOT/'IMPORT_ACCEPTANCE.json').open('x') as f:json.dump(result,f,indent=2)
    print(result)


if __name__=='__main__':main()
