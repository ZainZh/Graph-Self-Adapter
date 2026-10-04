"""P64 finite book cache from named, sealed P56 train/development episodes.

No environment construction or integration. Offline labels never enter public.pt.
"""
from pathlib import Path
import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
import numpy as np
import torch
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from torch.nn import functional as F
from prove.fast_adapter.p61_encoder import first_eef_anchor_transforms
from prove.fast_adapter.p64_book_bridge import BookMeasurement, visible_boundary
from prove.fast_adapter.visual_alignment import FrozenDinoROI, DEFAULT_MODEL

ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'evaluations/fast_adapter_p57_geometry_tokens_v01/geometry_bundle_v01'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def selected_indices(length,count):
    if length<count: raise ValueError('selected budget exceeds available frames')
    return np.linspace(0,length-1,count).round().astype(int).tolist()


def source_cases(source):
    manifest=json.loads((OLD/'MANIFEST.json').read_text())
    cases=[];counter={'train':0,'validation':0}
    counts={}
    for split,budget in (('train',256),('validation',32)):
        rows=[r for r in manifest['episodes'] if r['split']==split]
        for row in rows:counts[row['cell_index']]=0
        while budget:
            progressed=False
            for row in rows:
                key=row['cell_index']
                if budget and counts[key]<row['frames']:
                    counts[key]+=1;budget-=1;progressed=True
            if not progressed:raise ValueError('insufficient source frame capacity')
    for row in manifest['episodes']:
        split=row['split'];i=counter[split];counter[split]+=1
        pack_path=OLD/Path(row['bundle']['path']).name
        if sha(pack_path)!=row['bundle']['sha256']:raise ValueError('cached label pack changed')
        pack=torch.load(pack_path,weights_only=True)
        relative=row['source_seal']['path'].split('fast_adapter_p56_feedback_data_v01/')[1]
        episode=Path(source)/Path(relative).parent
        if sha(episode/'SEAL.json')!=row['source_seal']['sha256']:raise ValueError('source seal changed')
        seal=json.loads((episode/'SEAL.json').read_text())
        count=counts[row['cell_index']]
        for t in selected_indices(row['frames'],count):
            frame=episode/'public'/f'frame_{t:04d}.npz'
            if sha(frame)!=seal['files'][str(frame.relative_to(episode))]:raise ValueError('public pixels changed')
            cases.append(dict(split='train' if split=='train' else 'dev',cell_index=row['cell_index'],t=t,
                frame=frame,episode=episode,pack=pack,pack_path=pack_path,source_seal_sha256=sha(episode/'SEAL.json')))
    if counter!={'train':34,'validation':8} or len(cases)!=288:raise ValueError('book finite source budget differs')
    return cases


def visual_mesh(asset, frame_site=None):
    """Actual train asset mesh, retaining only its 3-D convex envelope vertices."""
    asset=Path(asset);xml=ET.parse(asset).getroot()
    geom=next(g for g in xml.iter('geom') if g.get('type')=='mesh' and g.get('contype')=='0')
    if geom.get('quat') or geom.get('euler'):raise ValueError('unimplemented visual geom rotation')
    mesh=next(m for m in xml.iter('mesh') if m.get('name')==geom.get('mesh'))
    path=asset.parent/mesh.attrib['file']
    with path.open('rb') as stream:
        counts=np.fromfile(stream,dtype='<i4',count=4)
        vertices=np.fromfile(stream,dtype='<f4',count=int(counts[0])*3).reshape(-1,3)
    scale=np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
    pos=np.fromstring(geom.get('pos','0 0 0'),sep=' ')
    vertices=vertices*scale+pos
    if frame_site is not None:
        site=next(v for v in xml.iter('site') if v.get('name')==frame_site)
        pos=np.fromstring(site.get('pos','0 0 0'),sep=' ')
        quat=np.fromstring(site.get('quat','1 0 0 0'),sep=' ')
        rotation=Rotation.from_quat(quat[[1,2,3,0]]).as_matrix()
        vertices=(vertices-pos)@rotation
    vertices=vertices[ConvexHull(vertices).vertices]
    return vertices,dict(xml=str(asset),xml_sha256=sha(asset),mesh=str(path),mesh_sha256=sha(path),
                         kind='actual_visual_mesh_3d_convex_envelope',frame_site=frame_site)


def priors_from_train(cases):
    train={c['cell_index']:c for c in cases if c['split']=='train'}
    original=next(c for c in train.values() if c['pack']['scale_xyz']==[1,1,1])
    poses=[c['pack']['labels'] for c in train.values()]
    mean_rotation=lambda key:Rotation.from_matrix(torch.cat([p[key][:,:3,:3] for p in poses]).numpy()).mean().as_matrix()
    calibration=json.loads((original['episode']/'SUPERVISOR_CALIBRATION.json').read_text())
    assets=ROOT/'data/sources/LIBERO-PRO/libero/libero/assets/turbosquid_objects'
    book,book_binding=visual_mesh(assets/'black_book/black_book.xml')
    caddy,caddy_binding=visual_mesh(assets/'desk_caddy/desk_caddy.xml',frame_site='back_contain_region')
    final=poses[0]['desired_book_in_caddy'][0].numpy()
    if not all(torch.allclose(p['desired_book_in_caddy'],torch.as_tensor(final).expand_as(p['desired_book_in_caddy']),atol=1e-5) for p in poses):
        raise ValueError('historical goal relation is not fixed')
    values=dict(book_vertices=book.tolist(),caddy_vertices=caddy.tolist(),
        book_scale_center=calibration['book_collision_center'],R_book_eef=mean_rotation('book_in_eef').tolist(),
        R_caddy_world=mean_rotation('world_caddy').tolist(),final_caddy_book=final.tolist())
    return dict(schema='prove.p64-book-train-priors.v1',values=values,
                assets=[book_binding,caddy_binding],train_cells=sorted(train),
                goal_size_semantics='fixed historical desired-book relation, no hidden test-scale adjustment',
                geometry9='book-in-EEF origin3,caddy-back-site-in-first-EEF-anchor origin3,log book scales3')


def frame_public(frame,anchor):
    from prove.task_graph.p57_public_geometry import eef_transform
    eef=torch.from_numpy(eef_transform(frame['eef_pos'],frame['eef_quat_xyzw'])).float()
    camera=torch.tensor(frame['camera_pose'],dtype=torch.float32)
    ac,ae=first_eef_anchor_transforms(anchor,camera,eef)
    masks=torch.tensor(frame['visible_masks'],dtype=torch.float32)
    return dict(role_patch_masks=F.avg_pool2d(masks.reshape(6,1,448,448),16,16).reshape(2,3,28,28),
                role_valid=masks.bool().flatten(-2).any(-1),K=torch.tensor(frame['camera_K'],dtype=torch.float32),
                T_anchor_camera=ac,T_anchor_eef=ae),eef


def measurements_for(case,anchor):
    pack=case['pack'];items=[]
    # Every actual observed frame up to query, two views, public sources only.
    for t in range(max(0,case['t']-7),case['t']+1):
        frame_path=case['episode']/'public'/f'frame_{t:04d}.npz'
        seal=json.loads((case['episode']/'SEAL.json').read_text())
        if sha(frame_path)!=seal['files'][str(frame_path.relative_to(case['episode']))]:raise ValueError('support public frame changed')
        with np.load(frame_path) as f:
            masks=f['visible_masks'].astype(bool)
            for view in range(2):
                for role in (1,2):
                    others=np.any(np.delete(masks[view],role,axis=0),axis=0)
                    boundary=visible_boundary(masks[view,role],others)
                    if boundary is None:continue
                    items.append(dict(role=role,boundary=torch.tensor(boundary),K=torch.tensor(f['camera_K'][view]),
                        camera_pose=torch.tensor(f['camera_pose'][view]),eef_transform=pack['input']['eef_transform'][t],
                        T_world_anchor=anchor,t=t))
    return items


def materialize(source,output,model=DEFAULT_MODEL,device='cuda'):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    cases=source_cases(source);prior=priors_from_train(cases)
    (output/'PRIORS.json').write_text(json.dumps(prior,indent=2)+'\n')
    net=FrozenDinoROI(model,device=device,microbatch=2);rows=[];means=[]
    for index,c in enumerate(cases):
        pack=c['pack'];t=c['t'];anchor=pack['input']['eef_transform'][0].float()
        with np.load(c['frame']) as frame:
            public,eef=frame_public(frame,anchor)
            with torch.inference_mode():
                pixels=torch.from_numpy(frame['rgb'].copy()).to(device).permute(0,3,1,2).float()/255
                pixels=((pixels-net.mean)/net.std).to(torch.bfloat16)
                hidden=net.model(pixel_values=pixels,output_hidden_states=True,return_dict=True).hidden_states
                public['dino_layers']=torch.stack([hidden[i][:,net.prefix:] for i in (-12,-8,-4)],1).cpu().to(torch.float16)
            public['T_world_anchor']=anchor
            public['eef_transform']=eef
            public['bboxwh']=(pack['input']['features']['boxes'][t,:,1,2:]-pack['input']['features']['boxes'][t,:,1,:2]).reshape(4)
            public['gripper']=torch.tensor(frame['gripper_qpos'],dtype=torch.float32)
        public['measurements']=measurements_for(c,anchor)
        center=(torch.linalg.inv(anchor)@pack['labels']['world_caddy'][t])[:3,3]
        label=torch.cat((pack['labels']['book_in_eef'][t,:3,3],center,torch.tensor(pack['scale_xyz']).float().log()))
        if c['split']=='train':means.append(label)
        folder=output/c['split']/f"cell_{c['cell_index']:03d}_t{t:03d}";folder.mkdir(parents=True)
        torch.save(public,folder/'public.pt');torch.save(dict(geometry=label,expert_action=(pack['expert_actions'][t] if t<len(pack['expert_actions']) else torch.zeros(7)),expert_action_valid=t<len(pack['expert_actions'])),folder/'private.pt')
        rows.append(dict(path=str(folder.resolve()),split=c['split'],cell_index=c['cell_index'],t=t,
            public_sha256=sha(folder/'public.pt'),private_sha256=sha(folder/'private.pt'),
            source_frame=str(c['frame']),source_frame_sha256=sha(c['frame']),source_pack_sha256=sha(c['pack_path']),source_seal_sha256=c['source_seal_sha256']))
        if index%16==0:print(json.dumps(dict(done=index+1,total=288)),flush=True)
    grouped={}
    for r in rows:grouped.setdefault((r['split'],r['cell_index']),[]).append(r)
    for group in grouped.values():
        group.sort(key=lambda r:r['t'])
        for i,r in enumerate(group):
            r['meta_support_paths']=[s['path'] for s in group[max(0,i-2):i]]
            r['meta_support_times']=[s['t'] for s in group[max(0,i-2):i]]
            r['meta_query_eligible']=i>=2
    manifest=dict(schema='prove.p64-book-cache.v1',records=rows,split_counts=dict(train=256,dev=32),
        geometry_mean=torch.stack(means).mean(0).tolist(),priors_sha256=sha(output/'PRIORS.json'),
        dino_weights_sha256=sha(Path(model)/'model.safetensors'),source_manifest_sha256=sha(OLD/'MANIFEST.json'),
        physics_steps=0,render_steps=0,source_code_sha256=sha(__file__))
    (output/'MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True)
    p.add_argument('--model',default=DEFAULT_MODEL);p.add_argument('--device',default='cuda')
    a=p.parse_args();materialize(a.source,a.output,a.model,a.device)
