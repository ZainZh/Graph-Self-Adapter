#!/usr/bin/env python3
"""File-transfer only; default prints plan. Never starts a controller or service."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import tarfile


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--package',type=Path,required=True);p.add_argument('--host',default='clover@100.126.63.65')
    p.add_argument('--destination',default='/home/clover/github/Graph-Self-Adapter/deployments/cup_direct_v15');p.add_argument('--transfer',action='store_true');a=p.parse_args(argv)
    package=a.package.resolve();archive=package.with_suffix('.tar.gz')
    if not archive.exists():
        with tarfile.open(archive,'w:gz') as tar:tar.add(package,arcname=package.name)
    sha=hashlib.sha256(archive.read_bytes()).hexdigest();remote='/tmp/'+archive.name;destination=shlex.quote(a.destination)
    commands=[['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',a.host,f'test ! -e {destination}'],
              ['scp','-o','BatchMode=yes','-o','ConnectTimeout=8',str(archive),a.host+':'+remote],
              ['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8',a.host,f'mkdir -p {destination} && tar -xzf {shlex.quote(remote)} --strip-components=1 -C {destination}']]
    plan={'archive':str(archive),'archive_sha256':sha,'host':a.host,'destination':a.destination,'commands':commands,'controller_started':False,'control_PID_touched':False,'camera_robot_connections':0,
        'filecheck_after_transfer':f'CUP_PYTHON=/home/clover/miniconda3/envs/geometry/bin/python bash {a.destination}/run.sh check',
        'real_motion_command':f'bash {a.destination}/run.sh real','default_launch_mode':'check; real explicitly executes same qualified runtime'}
    print(json.dumps(plan,indent=2),flush=True)
    if a.transfer:
        for command in commands:subprocess.run(command,check=True)
    return plan

if __name__=='__main__':main()
