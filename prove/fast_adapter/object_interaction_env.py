"""P29 parameterized physical objects, native Sawyer actuation, RGB-only public inputs.

Derived contact tasks, not official Meta-World benchmark scores. Geometry/contact
truth is confined to expert collection and evaluator; learned policies use public().
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

DEPS = Path('data/runtime/fast-adapter-p28-deps')
sys.path.insert(0, str(DEPS.resolve()))
import mujoco
from metaworld.sawyer_xyz_env import SawyerXYZEnv
from gymnasium.spaces import Box

TASKS = ('mug-handle-lift', 'hammer-block')
ACTOR_RGBA = '0.72 0.16 0.8 1'
TARGET_RGBA = '0.25 0.82 0.08 1'


def object_config(task, seed, split='train'):
    rng = np.random.default_rng(seed)
    if task == TASKS[0]:
        v = dict(radius=.034, height=.078, extension=.038, bar=.012, mass=.13, friction=1.1)
        ranges = dict(radius=(.028,.040), height=(.068,.092), extension=(.030,.046),
                      bar=(.0105,.014), mass=(.10,.18), friction=(.8,1.4))
        if split != 'canonical':
            v.update({k: float(rng.uniform(*ab)) for k,ab in ranges.items()})
        if split == 'geometry_ood':
            v['extension'] = float(rng.uniform(.050,.060))
            v['height'] = float(rng.uniform(.094,.108))
            v['radius'] = float(rng.uniform(.040,.046))
    else:
        v = dict(length=.15, shaft=.015, head=.033, head_width=.028, mass=.35, friction=1.1)
        ranges = dict(length=(.125,.175), shaft=(.012,.019), head=(.028,.039),
                      head_width=(.024,.035), mass=(.25,.50), friction=(.8,1.4))
        if split != 'canonical':
            v.update({k: float(rng.uniform(*ab)) for k,ab in ranges.items()})
        if split == 'geometry_ood':
            v['length'] = float(rng.uniform(.181,.21))
            v['head'] = float(rng.uniform(.040,.048))
    return {'task':task, 'object_seed':int(seed), 'split':split, **v}


def scene_xml(cfg):
    task = cfg['task']; geoms = []
    def geom(name, typ, pos=None, size=None, rgba=ACTOR_RGBA, **kw):
        attrs = dict(name=name, type=typ, rgba=rgba, group='0', contype='1', conaffinity='1',
                     condim='4', friction=f"{cfg['friction']} .01 .001", priority='1',
                     solref='.01 1', solimp='.95 .99 .001')
        if pos is not None: attrs['pos']=' '.join(map(str,pos))
        if size is not None:attrs['size']=' '.join(map(str,size))
        attrs.update(kw)
        geoms.append('<geom '+' '.join(f'{k}="{v}"' for k,v in attrs.items())+'/>')
    if task == TASKS[0]:
        r,h,g,b=cfg['radius'],cfg['height'],cfg['extension'],cfg['bar']
        geom('cup_base','cylinder',[0,0,.004],[r,.004])
        for i in range(16):
            theta=2*np.pi*i/16
            geom(f'cup_wall_{i}','box',[(r-.002)*np.cos(theta),(r-.002)*np.sin(theta),h/2],
                 [.004,r*np.sin(np.pi/16),h/2],euler=f'0 0 {theta}')
        x=r+g; lo=.018; hi=h*.82
        geom('actor_handle','capsule',size=[b],fromto=f'{x} 0 {lo} {x} 0 {hi}')
        geom('handle_bottom','capsule',size=[b],fromto=f'{r-.002} 0 {lo} {x} 0 {lo}')
        geom('handle_top','capsule',size=[b],fromto=f'{r-.002} 0 {hi} {x} 0 {hi}')
        sites=f'<site name="grip_site" pos="{x} 0 {(lo+hi)/2}" size=".002" rgba="0 0 0 0"/>'
        inertia=f'{cfg["mass"]*.002} {cfg["mass"]*.002} {cfg["mass"]*.0015}'
        inertial=f'<inertial pos=".008 0 {h/2}" mass="{cfg["mass"]}" diaginertia="{inertia}"/>'
        target=''; object_z=.009
    else:
        L,r,H,w=cfg['length'],cfg['shaft'],cfg['head'],cfg['head_width']
        geom('actor_handle','capsule',size=[r],fromto=f'{-L*.65} 0 0 {L*.72} 0 0')
        geom('actor_head','box',[L,0,0],[w,H,H])
        sites=f'<site name="grip_site" pos="{-L*.25} 0 0" size=".002" rgba="0 0 0 0"/><site name="tool_site" pos="{L} {H} 0" size=".002" rgba="0 0 0 0"/>'
        inertia=f'{cfg["mass"]*.002} {cfg["mass"]*.014} {cfg["mass"]*.014}'
        inertial=f'<inertial pos="{L*.55} 0 0" mass="{cfg["mass"]}" diaginertia="{inertia}"/>'
        object_z=H+.002
        target=f'''<body name="target" pos=".20 .74 .13">
          <joint name="target_slide" type="slide" axis="0 1 0" limited="true" range="0 .07" damping="3" frictionloss=".4"/>
          <inertial pos="0 0 0" mass=".2" diaginertia=".0002 .0002 .0002"/>
          <geom name="target_block" type="box" size=".032 .018 .032" rgba="{TARGET_RGBA}" contype="1" conaffinity="1" group="0"/>
          <site name="target_site" pos="0 -.018 0" size=".002" rgba="0 0 0 0"/>
        </body>
        <geom name="target_stand" type="box" pos=".20 .79 .047" size=".08 .10 .045" rgba=".43 .29 .17 1" contype="1" conaffinity="1" group="0"/>
        <geom name="target_track" type="box" pos=".20 .80 .093" size=".034 .09 .004" rgba=".34 .23 .14 1" contype="1" conaffinity="1" group="0"/>'''
    return f'''<mujoco model="p29_{task}">
      <include file="../scene/basic_scene.xml"/>
      <include file="../objects/assets/xyz_base_dependencies.xml"/>
      <worldbody><include file="../objects/assets/xyz_base.xml"/>
        <body name="actor" pos="-.07 .56 {object_z}"><freejoint name="actor_free"/>{inertial}{''.join(geoms)}{sites}</body>
        {target}
      </worldbody>
      <actuator><position ctrllimited="true" ctrlrange="-1 1" joint="r_close" kp="400" user="1"/><position ctrllimited="true" ctrlrange="-1 1" joint="l_close" kp="400" user="1"/></actuator>
      <equality><weld body1="mocap" body2="hand" solref=".02 1"/></equality>
    </mujoco>'''


def xml_path(cfg):
    import metaworld
    contents=scene_xml(cfg)
    name='p29_'+hashlib.sha256(contents.encode()).hexdigest()[:20]+'.xml'
    p=Path(metaworld.__file__).parent/'assets/sawyer_xyz'/name
    if not p.exists():p.write_text(contents)
    return str(p.resolve())


class _SawyerObjects(SawyerXYZEnv):
    def __init__(self,cfg):
        self.cfg=cfg;self._xml=xml_path(cfg)
        self.hold_streak=0;self.head_contact_seen=False;self.handle_contact_seen=False;self.direct_contact_seen=False
        super().__init__(hand_low=(-.4,.35,.045),hand_high=(.4,.95,.5),height=224,width=224)
        self.hand_init_pos=np.array([0,.4,.22])
        self._target_pos=np.array([0,.65,.20])
        self.goal_space=Box(np.array([-.4,.35,0]),np.array([.4,.95,.5]),dtype=np.float64)
        self._partially_observable=False
        if 'sawyer_observation_space' in self.__dict__:del self.sawyer_observation_space
        self._set_task_called=True
        self._freeze_rand_vec=False
        self.left=int(self.model.geom('leftpad_geom').id)
        self.right=int(self.model.geom('rightpad_geom').id)
        self.handle_geoms={int(self.model.geom('actor_handle').id)}
        if cfg['task']==TASKS[0]:self.handle_geoms.update(int(self.model.geom(k).id) for k in ('handle_bottom','handle_top'))
        self.body_geoms={i for i in range(self.model.ngeom) if self.model.geom_bodyid[i]==self.model.body('actor').id}
        self.start_z=0.

    @property
    def model_name(self):return self._xml

    @property
    def _target_site_config(self):return []

    def _get_pos_objects(self):return self.data.body('actor').xpos.copy()

    def _get_quat_objects(self):return self.data.body('actor').xquat.copy()

    def reset_model(self):
        self.set_state(self.init_qpos.copy(),self.init_qvel.copy())
        self._reset_hand()
        rng=self.np_random
        q=self.data.qpos.copy();v=np.zeros_like(self.data.qvel)
        j=int(self.model.joint('actor_free').qposadr[0])
        z=.009 if self.cfg['task']==TASKS[0] else self.cfg['head']+.002
        q[j:j+3]=[rng.uniform(-.12,.025),rng.uniform(.52,.63),z]
        q[j+3:j+7]=[1,0,0,0]
        if self.cfg['task']==TASKS[1]:
            self.model.body('target').pos[:2]=[rng.uniform(.15,.24),rng.uniform(.72,.79)]
            q[int(self.model.joint('target_slide').qposadr[0])]=0
        self.set_state(q,v)
        self.do_simulation([-1,1],40)
        mujoco.mj_forward(self.model,self.data)
        self.start_z=float(self.data.body('actor').xpos[2])
        self.hold_streak=0;self.head_contact_seen=False;self.handle_contact_seen=False;self.direct_contact_seen=False
        self.obj_init_pos=self.data.body('actor').xpos.copy()
        return self._get_obs()

    def contacts(self):
        pairs={frozenset((int(c.geom1),int(c.geom2))) for c in self.data.contact if c.dist<.002}
        left=any(frozenset((self.left,g)) in pairs for g in self.handle_geoms)
        right=any(frozenset((self.right,g)) in pairs for g in self.handle_geoms)
        body_touch=any(frozenset((f,g)) in pairs for f in (self.left,self.right) for g in self.body_geoms-self.handle_geoms)
        head=False;direct=False
        if self.cfg['task']==TASKS[1]:
            head=frozenset((int(self.model.geom('actor_head').id),int(self.model.geom('target_block').id))) in pairs
            target=int(self.model.geom('target_block').id)
            direct=any(target in p and any(int(self.model.geom_bodyid[g])!=self.model.body('actor').id and g!=target and int(self.model.geom_bodyid[g])!=0 for g in p) for p in pairs)
        return left,right,body_touch,head,direct

    def evaluate_state(self,obs,action):
        left,right,body,head,direct=self.contacts()
        lifted=float(self.data.body('actor').xpos[2]-self.start_z)
        self.handle_contact_seen |= left and right
        self.head_contact_seen |= head and left and right
        self.direct_contact_seen |= direct
        if self.cfg['task']==TASKS[0]:
            valid=left and right and lifted>=.07
            self.hold_streak=self.hold_streak+1 if valid else 0
            success=self.hold_streak>=15
            displacement=0.
        else:
            displacement=float(self.data.joint('target_slide').qpos[0])
            valid=self.head_contact_seen and displacement>=.045 and not self.direct_contact_seen
            self.hold_streak=self.hold_streak+1 if valid else 0
            success=self.hold_streak>=5
        return 0.,dict(success=float(success),left_handle=float(left),right_handle=float(right),
            bilateral_handle=float(left and right),body_touch=float(body),head_target_contact=float(head),
            target_direct_contact=float(direct),lift=lifted,target_displacement=displacement,
            hold_streak=self.hold_streak,head_contact_seen=float(self.head_contact_seen))


class ObjectEnv:
    def __init__(self,cfg,render=True):
        self.cfg=dict(cfg);self.task=cfg['task'];self.env=_SawyerObjects(cfg)
        self.renderer=mujoco.Renderer(self.env.model,224,224) if render else None
        self.cam=mujoco.MjvCamera();self.cam.type=mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:]=[0,.65,.10];self.cam.distance=.95;self.cam.azimuth=140;self.cam.elevation=-32
        self.metrics=[];self.phase=0;self.wait=0
    def reset(self,seed):
        self.env.seed(int(seed));self.obs,_=self.env.reset();self.metrics=[];self.phase=0;self.wait=0
        return self.public()
    def public(self):return self.obs[:4].astype(np.float32).copy()
    def rgb(self):
        self.renderer.update_scene(self.env.data,camera=self.cam)
        return self.renderer.render().copy()
    def step(self,action):
        a=np.clip(np.asarray(action,dtype=np.float64),-1,1)
        self.obs,_,_,_,info=self.env.step(a)
        self.metrics.append({k:float(v) for k,v in info.items()})
        return self.public()
    def teacher(self):
        e=self.env;p=e.tcp_center.copy();g=e.data.site('grip_site').xpos.copy()
        # Only training teacher reads true sites and phase. Learned runtime never does.
        grasp=g.copy();grasp[2]-=.020 if self.task==TASKS[0] else .005
        if self.phase==0:
            goal=grasp.copy();goal[2]=max(grasp[2]+.12,.19);grip=-1.
            if np.linalg.norm((goal-p)[:2])<.009:self.phase=1
        elif self.phase==1:
            goal=grasp.copy();grip=-1.
            if np.linalg.norm(goal-p)<.010:self.phase=2;self.wait=0
        elif self.phase==2:
            goal=grasp.copy();grip=.8;self.wait+=1
            if self.wait>=12:self.phase=3
        elif self.task==TASKS[0]:
            goal=grasp.copy();goal[2]=e.start_z+.18;grip=.8
        else:
            grip=.9
            tip=e.data.site('tool_site').xpos.copy();target=e.data.site('target_site').xpos.copy()
            if self.phase==3:
                desired=target.copy();desired[1]-=.10
                goal=p+(desired-tip)
                if np.linalg.norm((desired-tip)[[0,2]])<.013 and abs(desired[1]-tip[1])<.025:self.phase=4
            else:
                desired=target.copy();desired[1]+=.061-float(e.data.joint('target_slide').qpos[0])
                goal=p+(desired-tip)
        return np.r_[np.clip((goal-p)*15,-1,1),grip].astype(np.float32)
    def state(self):
        return {k:getattr(self.env.data,k).copy() for k in ('qpos','qvel','mocap_pos','mocap_quat')}
    def close(self):
        if self.renderer is not None:self.renderer.close()
        self.env.close()
