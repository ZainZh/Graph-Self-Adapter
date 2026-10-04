#!/usr/bin/env python3
"""P58 hammer B6x6/z0 train-only geometry meta-training."""
from __future__ import annotations

import argparse,hashlib,json
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch

from prove.fast_adapter.p58_geometry_model import GeometryAdapterConfig,LowRankRayAdapter
from prove.fast_adapter.p58_hammer_readout import *
from tools.fast_adapter.p58_hammer_readout import load
from tools.fast_adapter.p58_hammer_freeze import verify as verify_freeze


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def samples(manifest_path,readout_path,representation):
    _,rows=load(manifest_path);rows=[r for r in rows if r["split"]=="train"]
    model=load_package(readout_path);result=[]
    for object_id in range(24):
        for role in ("expert","base"):
            episode=[r for r in rows if r["object_id"]==object_id and r["role"]==role]
            systems=[]
            estimates=[]
            for row in episode:
                uv,estimate=predict_measurement(model,row["frame"])
                # Both representations enter the identical canonical geometry:
                # hammer head in current EEF, target in world.  In particular,
                # a moving head-world estimate is never treated as a static ray.
                if representation=="absolute_positions":
                    eef=row["frame"]["eef_transform"]
                    estimate=estimate.copy();estimate[0]=eef[:3,:3].T@(estimate[0]-eef[:3,3])
                systems.append(ray_system(uv,row["frame"],"role_relative"))
                estimates.append(estimate)
            prefix=0
            for row in episode:
                if row["contact"] or abs(row["displacement"])>.001:break
                prefix+=1
            for end in range(2,prefix):
                A=np.stack([systems[t][0] for t in range(max(0,end-7),end+1)])
                C=np.stack([systems[t][1] for t in range(max(0,end-7),end+1)])
                valid=np.stack([systems[t][2] for t in range(max(0,end-7),end+1)])
                g=estimates[end]
                truth=episode[end]["relative"].reshape(2,3)
                result.append(dict(g=g,A=A,C=C,valid=valid,truth=truth,object_id=object_id,role=role,t=episode[end]["t"]))
    if not result:raise ValueError("No observable static-prefix hammer samples")
    return result


def batch(samples,indices):
    n=len(indices);A=np.zeros((n,16,2,3,3),np.float32);C=np.zeros((n,16,2,3),np.float32);valid=np.zeros((n,16,2),bool)
    for j,index in enumerate(indices):
        sample=samples[index]
        # Stored systems are [time,role,view,...].  Flatten time*view into
        # support while retaining the two semantic points as point dimension.
        aa=sample["A"].transpose(0,2,1,3,4).reshape(-1,2,3,3)
        cc=sample["C"].transpose(0,2,1,3).reshape(-1,2,3)
        vv=sample["valid"].transpose(0,2,1).reshape(-1,2)
        length=len(aa);A[j,-length:]=aa;C[j,-length:]=cc;valid[j,-length:]=vv
    tensor=lambda x:torch.from_numpy(np.asarray(x))
    return (tensor(np.stack([samples[i]["g"] for i in indices])).float(),tensor(A),tensor(C),tensor(valid),
            tensor(np.stack([samples[i]["truth"] for i in indices])).float())


def train(manifest,freeze,readout_path,representation,output):
    frozen=verify_freeze(freeze,manifest)
    if representation not in REPRESENTATIONS:raise ValueError("Unknown representation")
    report_path=Path(readout_path).with_name("READOUT_REPORT.json")
    report=json.loads(report_path.read_text())
    package=report.get("packages",{}).get(representation,{})
    if (report.get("schema")!="p58.hammer-ridge-report.v1" or report.get("manifest_sha256")!=sha(manifest)
            or report.get("freeze_sha256")!=sha(freeze) or package.get("sha256")!=sha(readout_path)
            or Path(package.get("path","")).resolve()!=Path(readout_path).resolve()):
        raise ValueError("P58 hammer readout report/package binding mismatch")
    data=samples(manifest,readout_path,representation)
    residual=np.stack([(x["truth"]-x["g"]).reshape(-1) for x in data])/.01
    _,_,vt=np.linalg.svd(residual-residual.mean(0),full_matrices=False);basis=vt[:6].T.astype(np.float32)
    config=GeometryAdapterConfig(point_count=2,latent_dim=6,inner_steps=3,inner_lr=.65,damping=1e-4,step_cap=1.,geometry_scale=.01)
    torch.manual_seed(11);rng=np.random.default_rng(11);model=LowRankRayAdapter(basis,config)
    optimizer=torch.optim.Adam(model.parameters(),lr=3e-4);trace=[]
    for step in range(500):
        indices=rng.integers(0,len(data),size=16);g,A,C,valid,truth=batch(data,indices)
        z0=model.z0[None].expand(len(g),-1);z,_=model.adapt(g,z0,A,C,valid,create_graph=True)
        fixed=((model.decode(g,z0)-truth)/.01).square().mean();online=((model.decode(g,z)-truth)/.01).square().mean()
        loss=.5*(fixed+online);optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if (step+1)%50==0:
            trace.append(dict(step=step+1,loss=float(loss.detach()),fixed=float(fixed.detach()),online=float(online.detach())))
            print(representation,trace[-1],flush=True)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists() or output.with_suffix(".json").exists():raise FileExistsError(output)
    package=dict(schema="p58.hammer-geometry-model.v1",representation=representation,config=asdict(config),
      state_dict=model.state_dict(),manifest_sha256=sha(manifest),freeze_sha256=sha(freeze),readout_sha256=sha(readout_path),
      training_samples=len(data),training_objects=24,canonical_geometry="head_in_eef_plus_target_world",
      training_mean_canonical=torch.from_numpy(np.stack([x["truth"] for x in data]).mean(0).astype(np.float32)),trace=trace)
    torch.save(package,output);report=dict(schema="p58.hammer-geometry-training-report.v1",representation=representation,
      checkpoint=str(output),checkpoint_sha256=sha(output),training_samples=len(data),training_objects=24,trace=trace)
    output.with_suffix(".json").write_text(json.dumps(report,indent=2)+"\n");print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--freeze",type=Path,required=True)
    p.add_argument("--readout",type=Path,required=True);p.add_argument("--representation",choices=REPRESENTATIONS,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args();train(a.manifest,a.freeze,a.readout,a.representation,a.output)
