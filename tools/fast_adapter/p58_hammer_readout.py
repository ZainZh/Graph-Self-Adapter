#!/usr/bin/env python3
"""Grouped train-only ridge selection for P58 held-hammer semantic centers."""
from __future__ import annotations

import argparse, hashlib, json
from pathlib import Path
import numpy as np

from prove.fast_adapter.p58_hammer_readout import *
from tools.fast_adapter.p58_hammer_freeze import verify as verify_freeze


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj): Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False)+"\n")


def load(manifest_path):
    manifest=json.loads(Path(manifest_path).read_text());root=Path(manifest_path).parent;rows=[]
    for item in manifest["rows"]:
        public=np.load(root/item["public_path"],allow_pickle=False)
        private=np.load(root/item["private_path"],allow_pickle=False)
        for index,t in enumerate(public["times"]):
            frame={key:public[key][index] for key in public.files if key!="times"}
            y,valid=relative_uv(frame,private["semantic_uv"][index])
            rows.append(dict(object_id=item["object_id"],role=item["role"],split=item["split"],t=int(t),
                x=public_features(frame),y=y,valid=valid,frame=frame,
                absolute=np.asarray(private["absolute_positions"][index]),
                relative=np.asarray(private["role_relative"][index]),
                displacement=float(private["target_displacement"][index]),
                contact=bool(private["head_target_contact"][index])))
    return manifest,rows


def model_for(train,dim,strength,target="y"):
    x=np.stack([r["x"] for r in train]);y=np.stack([r[target] for r in train])
    valid=(np.stack([r["valid"] for r in train]) if target=="y" else np.ones_like(y,dtype=bool))
    pca=fit_pca(x,dim);score=transform_pca(pca,x);coef=fit_ridge(score,y,valid,strength)
    return dict(pca=pca,coef=coef)


def score_uv(model,rows):
    pixel=[]
    for row in rows:
        predicted=pixels_from_relative(predict_relative(model,row["x"][None])[0],row["frame"])
        truth=pixels_from_relative(row["y"],row["frame"])
        mask=row["valid"].reshape(2,2,2)[...,0]
        pixel.extend(np.linalg.norm(predicted-truth,axis=-1)[mask].tolist())
    return dict(pixel_mean=float(np.mean(pixel)),pixel_p95=float(np.quantile(pixel,.95)),frames=len(rows))


def score_xyz(model,rows,target):
    errors=[]
    for row in rows:
        predicted=predict_relative(model,row["x"][None])[0].reshape(2,3)
        truth=np.asarray(row[target]).reshape(2,3)
        errors.extend((np.linalg.norm(predicted-truth,axis=-1)*1000).tolist())
    return dict(position_mm_mean=float(np.mean(errors)),position_mm_p95=float(np.quantile(errors,.95)),frames=len(rows))


def package_score(uv_model,xyz_model,rows,target):
    return score_uv(uv_model,rows)|score_xyz(xyz_model,rows,target)


def main(manifest_path,freeze_path,output):
    freeze=verify_freeze(freeze_path,manifest_path)
    manifest,rows=load(manifest_path);train=[r for r in rows if r["split"]=="train"]
    dev=[r for r in rows if r["split"]=="development"]
    uv_candidates=[]
    for dim in PCA_DIMS:
        for strength in RIDGE:
            fold_scores=[]
            for fold in range(5):
                fit=[r for r in train if r["object_id"]%5!=fold]
                held=[r for r in train if r["object_id"]%5==fold]
                model=model_for(fit,dim,strength,"y")
                fold_scores.append(score_uv(model,held)["pixel_mean"])
            uv_candidates.append(dict(dim=dim,strength=strength,fold_pixel_mean=fold_scores,
                                   cv_pixel_mean=float(np.mean(fold_scores))))
    uv_selected=min(uv_candidates,key=lambda r:(r["cv_pixel_mean"],r["dim"],r["strength"]))
    uv_model=model_for(train,uv_selected["dim"],uv_selected["strength"],"y")
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    static_prefix=[]
    for item in manifest["rows"]:
        episode=[r for r in rows if r["object_id"]==item["object_id"] and r["role"]==item["role"]]
        count=0
        for row in episode:
            if row["contact"] or abs(row["displacement"])>.001:break
            count+=1
        static_prefix.append(dict(object_id=item["object_id"],role=item["role"],frames=count,
                                  usable=count>=3,reason=None if count>=3 else "fewer_than_3_precontact_static_frames"))
    packages={}
    for representation in REPRESENTATIONS:
        target="absolute" if representation=="absolute_positions" else "relative"
        xyz_candidates=[]
        for dim in PCA_DIMS:
            for strength in RIDGE:
                fold_xyz=[]
                for fold in range(5):
                    fit=[r for r in train if r["object_id"]%5!=fold]
                    held=[r for r in train if r["object_id"]%5==fold]
                    xyz=model_for(fit,dim,strength,target)
                    fold_xyz.append(score_xyz(xyz,held,target)["position_mm_mean"])
                xyz_mean=float(np.mean(fold_xyz))
                xyz_candidates.append(dict(dim=dim,strength=strength,fold_position_mm_mean=fold_xyz,
                    cv_position_mm_mean=xyz_mean,
                    cv_joint_score=.5*(uv_selected["cv_pixel_mean"]/5.+xyz_mean/5.)))
        xyz_selected=min(xyz_candidates,key=lambda r:(r["cv_joint_score"],r["dim"],r["strength"]))
        xyz_model=model_for(train,xyz_selected["dim"],xyz_selected["strength"],target)
        path=output.with_name(output.stem+"_"+representation+output.suffix)
        if path.exists():raise FileExistsError(path)
        np.savez_compressed(path,
            uv_mean=uv_model["pca"]["mean"],uv_scale=uv_model["pca"]["scale"],
            uv_components=uv_model["pca"]["components"],uv_score_scale=uv_model["pca"]["score_scale"],uv_coef=uv_model["coef"],
            xyz_mean=xyz_model["pca"]["mean"],xyz_scale=xyz_model["pca"]["scale"],
            xyz_components=xyz_model["pca"]["components"],xyz_score_scale=xyz_model["pca"]["score_scale"],xyz_coef=xyz_model["coef"])
        packages[representation]=dict(path=str(path),sha256=sha(path),xyz_candidates=xyz_candidates,
            xyz_selected=xyz_selected,development=package_score(uv_model,xyz_model,dev,target))
    report=dict(schema="p58.hammer-ridge-report.v1",manifest_sha256=sha(manifest_path),
        train_frames=len(train),development_frames=len(dev),training_objects=24,development_objects=8,
        paired_roles_not_split_across_folds=True,uv_candidates=uv_candidates,uv_selected=uv_selected,
        static_prefix=static_prefix,usable_static_prefixes=sum(x["usable"] for x in static_prefix),
        packages=packages,development_not_used_for_selection=True,
        development_material_previously_opened=True,not_fresh_heldout=True,freeze_sha256=sha(freeze_path))
    report_path=output.with_name("READOUT_REPORT.json")
    if report_path.exists():raise FileExistsError(report_path)
    dump(report_path,report);print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--manifest",type=Path,required=True);p.add_argument("--freeze",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();main(a.manifest,a.freeze,a.output)
