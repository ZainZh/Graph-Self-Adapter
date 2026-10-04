"""P51 mixed full-task controller: code motion and frozen demonstration action MLP.

Only public EEF/gripper state enters runtime. The initial nominal target reference
comes from the opened teacher; set_reference supports a later visual producer.
No runtime simulator object pose, contact or success labels are accepted.
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from .libero_demo_skill import FrozenDemoSkill, QualifierRuleRouter

PRECISE = {'s3': 0, 's6': 1, 's7': 2}


class PrecisionMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(15, 96), torch.nn.Tanh(),
                    torch.nn.Linear(96,96),torch.nn.Tanh(),torch.nn.Linear(96,6))
    def forward(self,x):
        return self.net(x)


def features(position, rotation, gripper, reference, progress, stage):
    p,R=reference[:3,3],reference[:3,:3]
    onehot=np.eye(3)[PRECISE[stage]]
    return np.r_[R.T@(position-p)/.3,(R.T@rotation)[:,:2].reshape(-1),
                 np.asarray(gripper)/.04,progress,onehot].astype(np.float32)


def project_progress(position, rotation, positions, rotations, floor=0.):
    """Monotone projection onto the pose polyline; never uses action count/time."""
    rotations=np.asarray(rotations)
    vectors=Rotation.from_matrix(rotations @ rotation.T).as_rotvec()
    distance=np.sum((positions-position)**2,axis=1)+.02**2*np.sum(vectors**2,axis=1)
    n=len(positions)
    eligible=np.arange(n)>=int(np.floor(floor*max(n-1,1)))
    index=int(np.argmin(np.where(eligible,distance,np.inf)))
    candidates=[float(index)]
    for left in (index-1,index):
        if 0<=left<n-1:
            delta=positions[left+1]-positions[left]
            if delta@delta>1e-12:
                alpha=np.clip((position-positions[left])@delta/(delta@delta),0,1)
                if left+alpha>=floor*max(n-1,1):candidates.append(left+float(alpha))
    # Closest pose sample anchors projection; adjacent position interpolation adds continuity.
    projected=min(candidates,key=lambda u: np.linalg.norm(position-((1-(u%1))*positions[int(u)]+(u%1)*positions[min(int(u)+1,n-1)])))
    return max(floor,projected/max(n-1,1))


class MixedRouter(QualifierRuleRouter):
    @staticmethod
    def required_kind(skill):
        return 'frozen_neural' if skill in ('Grasp','Align','Insert') else 'code'


class MixedController:
    revision='p51-demo-action-mlp-and-code-v1'
    def __init__(self, source, graph, checkpoint):
        self.code=FrozenDemoSkill.from_source(source,graph)
        self.segments=self.code.segments
        package=torch.load(checkpoint,map_location='cpu',weights_only=False)
        self.model=PrecisionMLP();self.model.load_state_dict(package['state_dict'])
        self.model.eval().requires_grad_(False)
        self.reference=np.asarray(package['nominal_reference'],float)
        self.nominal_reference=self.reference.copy()
        self.paths=package['paths']
        self.stage_id=None;self.progress=0.;self.diagnostics={}
        self.reference_source='nominal_opened_demo_reference_adapter_disabled'
    def set_reference(self, transform, *, source):
        T=np.asarray(transform,float)
        if T.shape!=(4,4) or not np.isfinite(T).all() or not np.allclose(T[3],[0,0,0,1]) or not np.allclose(T[:3,:3].T@T[:3,:3],np.eye(3),atol=1e-6) or np.linalg.det(T[:3,:3])<.999:
            raise ValueError('finite rigid reference transform required')
        if source not in ('nominal_opened_demo_reference_adapter_disabled','public_visual_adapter'):
            raise ValueError('forbidden runtime reference authority')
        self.reference=T.copy();self.reference_source=source
    def set_observation(self, obs):
        self.code.set_observation(obs)
        self.position=np.asarray(obs['robot0_eef_pos'],float).copy()
        self.rotation=Rotation.from_quat(obs['robot0_eef_quat']).as_matrix()
        self.gripper=np.asarray(obs['robot0_gripper_qpos'],float).copy()
        if self.gripper.shape!=(2,) or not np.isfinite(self.gripper).all():raise ValueError('invalid gripper state')
    def on_enter(self, stage_id):
        self.stage_id=stage_id;self.progress=0.;self.code.on_enter(stage_id)
    def on_exit(self):
        self.stage_id=None;self.code.on_exit()
    def __call__(self,payload):
        stage=payload['stage_id']
        if stage!=self.stage_id:raise ValueError('stage entry required')
        if stage not in PRECISE:
            action=self.code(payload);self.diagnostics=dict(self.code.diagnostics,controller='code_eef_feedback')
            return action
        path=self.paths[stage]
        # Transform frozen nominal paths with the current public reference estimate.
        reference = self.nominal_reference if stage == 's3' else self.reference
        relative=reference@np.linalg.inv(self.nominal_reference)
        positions=np.asarray(path['positions'])@relative[:3,:3].T+relative[:3,3]
        rotations=relative[:3,:3]@np.asarray(path['rotations'])
        self.progress=project_progress(self.position,self.rotation,positions,rotations,self.progress)
        x=features(self.position,self.rotation,self.gripper,reference,self.progress,stage)
        with torch.no_grad():local=self.model(torch.from_numpy(x)).numpy()
        R=reference[:3,:3]
        action=np.r_[R@local[:3],R@local[3:],1.]
        action=np.clip(action,-1.,1.)
        self.diagnostics=dict(controller='frozen_neural_action',progress=float(self.progress),features=x.tolist(),
                              local_action=local.tolist(),reference_source='nominal_book_grasp_reference' if stage == 's3' else self.reference_source,
                              adapter_enabled=stage in ('s6','s7') and self.reference_source == 'public_visual_adapter')
        return action
