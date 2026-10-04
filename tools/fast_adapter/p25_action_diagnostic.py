"""Post-result common-state diagnostic, never consumed by a runtime controller."""
import json
import numpy as np
import torch
from prove.fast_adapter.geometry_insert import GeometryInsert, geometry, domain
from tools.fast_adapter.p25_common import R, SEEDS, load_adapter
from tools.fast_adapter.p18_run import digest, save_json


def main():
    torch.set_num_threads(2); torch.backends.cudnn.enabled=False
    rows=[];bindings=[]
    for st in ('id','position','scale'):
        root=R/'development'/st;rawpath=root/'base_RAW.pt';raw=torch.load(rawpath,weights_only=False)
        v=raw['visual'];p=torch.tensor(raw['positions'],dtype=torch.float32);a=torch.tensor(raw['actions'])
        prev=torch.cat((torch.zeros_like(a[:,:1]),a[:,:-1]),1);teacher=[]
        for i in range(len(p)):
            e=GeometryInsert(geometry(raw['seed_base']+i,st),render=False)
            e.set_domain(*domain(raw['seed_base']+100000+i));values=[]
            for t in range(48):
                e.data.qpos[:]=raw['positions'][i,t];values.append(e.expert_action())
            teacher.append(values);e.close()
        teacher=torch.tensor(np.asarray(teacher),dtype=torch.float32)
        reachable=torch.maximum(torch.minimum(teacher,(a+.35).clamp(-1,1)),(a-.35).clamp(-1,1))
        base_error=(a-reachable).square().mean(-1)
        for seed in SEEDS:
            net=load_adapter(seed,'physical','cpu');latentpath=root/f'LATENTS_seed{seed}.pt'
            lat=torch.load(latentpath,weights_only=True)['latents']['physical']
            with torch.no_grad():
                u=net.encode(v[:,:48],p[:,:48],prev);target=(p[:,1:]-p[:,:48])/.008
                entry={'stratum':st,'seed':seed,'base_feasible_teacher_mse':float(base_error.mean()),'modes':{}}
                for mode,z in {'k0':net.initial(len(p)),**lat}.items():
                    z=z[:,None].expand(-1,48,-1)
                    predicted=net.signal_prediction(u,a,z);action=net.action_u(u,a,z)
                    error=(action-reachable).square().mean(-1)
                    entry['modes'][mode]={'next_displacement_mse':float((predicted-target).square().mean()),
                        'action_feasible_teacher_mse':float(error.mean()),
                        'fraction_base_states_action_error_reduced':float((error<base_error-1e-8).float().mean()),
                        'fraction_base_states_action_error_increased':float((error>base_error+1e-8).float().mean())}
            rows.append(entry);bindings.append({'path':str(latentpath),'sha256':digest(latentpath)})
        bindings.append({'path':str(rawpath),'sha256':digest(rawpath)})
    report={'scope':'Posthoc diagnostic after all development outcomes; evaluate frozen physical heads on the exact same recorded base-policy states and actions. No new rollout, training, model selection or runtime ground-truth input.',
        'truth_access':'Evaluator-only geometry/dynamics used to compute action reference after trajectory seal. Projection into base +/-0.35 is a feasible action label, not a proven globally optimal control.',
        'rows':rows,'bindings':bindings}
    save_json(R/'summary/POSTHOC_COMMON_STATE_DIAGNOSTIC.json',report)


if __name__=='__main__':main()
