"""Zero-render, zero-physics train-only actual mesh and camera sanity audit."""
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from tools.fast_adapter.p64_book_data import source_cases,priors_from_train,sha
from prove.task_graph.p57_public_geometry import project_world_points
from prove.fast_adapter.p60_contour import project_world_points as torch_project


def audit(source,output):
    cases=source_cases(source);prior=priors_from_train(cases)['values'];records=[]
    for c in cases:
        if c['split']!='train':continue
        with np.load(c['frame']) as f:
            for role,key,name in ((1,'world_book','book'),(2,'world_caddy','caddy')):
                vertices=np.array(prior[name+'_vertices'])
                if role==1:
                    center=np.array(prior['book_scale_center'])
                    vertices=center+(vertices-center)*np.array(c['pack']['scale_xyz'])
                pose=c['pack']['labels'][key][c['t']].numpy()
                world=vertices@pose[:3,:3].T+pose[:3,3]
                for view in range(2):
                    K=f['camera_K'][view];camera=f['camera_pose'][view]
                    uv,front=project_world_points(world,K,camera)
                    other=torch_project(torch.tensor(world),torch.tensor(K),torch.tensor(camera[:3,3]),torch.tensor(camera[:3,:3])).numpy()
                    mask=np.zeros((448,448),np.uint8)
                    cv2.fillConvexPoly(mask,cv2.convexHull(uv.astype(np.float32)).astype(np.int32),1)
                    visible=f['visible_masks'][view,role].astype(bool)
                    records.append(dict(cell_index=c['cell_index'],t=c['t'],role=role,view=view,
                        all_vertices_in_front=bool(front.all()),camera_projection_max_delta=float(np.max(np.abs(uv-other))),
                        visible_pixels=int(visible.sum()),visible_mask_coverage=float(mask[visible].mean()) if visible.any() else None))
    valid=[r['visible_mask_coverage'] for r in records if r['visible_mask_coverage'] is not None]
    result=dict(schema='prove.p64-book-projection-audit.v1',source='256 selected train frames only; labels used only for coordinate sanity',
        physics_steps=0,render_steps=0,records=records,frames=256,mask_views=len(records),
        min_visible_mask_coverage=min(valid),mean_visible_mask_coverage=float(np.mean(valid)),
        all_vertices_in_front=all(r['all_vertices_in_front'] for r in records),
        max_camera_projection_delta=max(r['camera_projection_max_delta'] for r in records),
        caddy_frame='back_contain_region site: actual mesh transformed from body via train XML',
        public_projection='OpenGL negative z; already top-left images, no RGB/mask flip',
        priors=prior,source_code_sha256=sha(__file__))
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    return {k:v for k,v in result.items() if k not in ('records','priors')}

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    print(json.dumps(audit(a.source,a.output),indent=2))
