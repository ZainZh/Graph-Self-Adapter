"""P40 engineering near-contact skills, with visible-mask observations.

These are custom fixtures, not benchmark results. Cup is a free rigid body;
hammer and peg use translational tool stages. Runtime cannot read object truth.
The calibrated side view measures x/z assuming the interaction plane y=0.
"""
import numpy as np

DT = .04
STEPS = 100
TASKS = ('cup', 'hammer', 'insert')


def configuration(task, seed, split='train'):
    r = np.random.default_rng(seed)
    c = dict(task=task, seed=int(seed), split=split,
             mass=float(r.uniform(.06, .10)), friction=float(r.uniform(.35, .65)),
             resistance=float(r.uniform(.15, .35)), damping=float(r.uniform(3., 5.)),
             size=float(r.uniform(.9, 1.1)), offset=float(r.uniform(-.004, .004)),
             colors=r.uniform(.2, .9, (3, 3)).tolist())
    if split in ('physical', 'joint'):
        c.update(mass=float(r.uniform(.12, .17)), friction=float(r.uniform(.08, .18)),
                 resistance=float(r.uniform(.45, .7)), damping=float(r.uniform(7., 11.)))
    if task == 'insert':
        c['friction'] = float(r.uniform(.8, 1.1) if split in ('physical', 'joint') else r.uniform(.2, .4))
    if split in ('geometry', 'joint'):
        c.update(size=float(r.uniform(1.15, 1.3)), offset=float(r.uniform(.006, .010)))
    return c


def _rgba(v):
    return ' '.join(map(str, v)) + ' 1'


class ContactSkill:
    def __init__(self, cfg, render=True, resolution=448):
        import mujoco
        self.mj = mujoco; self.cfg = dict(cfg); self.task = cfg['task']
        self.resolution = resolution; self.time = 0
        s = cfg['size']; off = cfg['offset']; a, b, g = map(_rgba, cfg['colors'])
        # Fixed, externally calibrated side camera. No target truth lookup.
        camera = '<camera name="side" pos=".025 -.45 .075" xyaxes="1 0 0 0 0 1" fovy="35"/>'
        common = f'''<mujoco model="p40_{self.task}"><compiler angle="radian"/>
        <option timestep=".001" integrator="implicitfast" gravity="0 0 -9.81" iterations="80" cone="elliptic"/>
        <visual><global offwidth="448" offheight="448"/><quality offsamples="0" shadowsize="1024"/></visual>
        <default><geom condim="4" friction=".5 .015 .002" solref=".004 1" solimp=".95 .99 .001"/>
        <joint damping=".2"/></default><worldbody>{camera}<light pos="0 -.2 .5"/>
        <geom name="table" type="plane" size=".25 .25 .01" rgba=".23 .27 .3 1"/>'''
        if self.task == 'cup':
            radius=.018*s; height=.045*s; hx=off+radius+.019*s; hz=.025*s
            body=[f'<body name="object" pos="{off} 0 .001"><freejoint name="object_free"/>'
                  f'<inertial pos=".004 0 {height/2}" mass="{cfg["mass"]}" diaginertia=".00007 .0001 .00007"/>'
                  f'<geom name="object_base" type="cylinder" pos="0 0 .003" size="{radius} .003" rgba="{b}"/>']
            for i in range(12):
                th=i*np.pi/6
                body.append(f'<geom name="object_wall{i}" type="box" pos="{(radius-.0015)*np.cos(th)} {(radius-.0015)*np.sin(th)} {height/2}" size=".002 {radius*np.sin(np.pi/12)} {height/2}" euler="0 0 {th}" rgba="{b}"/>')
            body.append(f'<geom name="part_handle" type="box" pos="{radius+.019*s} 0 {hz}" size=".004 .005 {height*.31}" friction="{cfg["friction"]} .03 .004" priority="1" rgba="{g}"/>')
            for z in (hz-height*.28, hz+height*.28):
                body.append(f'<geom name="part_bridge{z}" type="box" pos="{radius+.008*s} 0 {z}" size="{.011*s} .005 .003" rgba="{g}"/>')
            body.append('</body>')
            tool=f'''<body name="tool" pos="{hx} 0 {hz+.001}">
            <joint name="tool_x" type="slide" axis="1 0 0"/><joint name="tool_z" type="slide" axis="0 0 1"/>
            <geom name="wrist" type="box" pos="0 0 .03" size=".010 .025 .004" rgba="{a}" contype="0" conaffinity="0"/>
            <body name="left" pos="0 -.012 0"><joint name="left_jaw" type="slide" axis="0 1 0" limited="true" range="0 .012" damping="1"/>
            <geom name="actor_left" type="box" size=".008 .003 .012" mass=".015" rgba="{a}"/></body>
            <body name="right" pos="0 .012 0"><joint name="right_jaw" type="slide" axis="0 -1 0" limited="true" range="0 .012" damping="1"/>
            <geom name="actor_right" type="box" size=".008 .003 .012" mass=".015" rgba="{a}"/></body></body>'''
            world=''.join(body)+tool
            actuator='<position joint="tool_x" kp="1800" kv="35"/><position joint="tool_z" kp="1800" kv="35"/><motor joint="left_jaw"/><motor joint="right_jaw"/>'
            self.initial_actor=np.array([hx,hz+.001]);self.lift_goal=.035
        elif self.task == 'hammer':
            top=.065*s; hh=.008*s; x=off
            world=f'''<body name="object" pos="{x} 0 {top-.009}">
            <joint name="object_slide" type="slide" axis="0 0 -1" limited="true" range="0 .045" damping="{cfg['damping']}" frictionloss="{cfg['resistance']}"/>
            <geom name="object_block" type="box" size="{.014*s} .015 .009" mass="{cfg['mass']}" rgba="{b}"/></body>
            <geom name="fixture" type="box" pos="{x} .023 .02" size=".027 .004 .020" rgba=".45 .3 .15 1"/>
            <geom name="goal_marker" type="box" pos="{x+.034} 0 {top-.018}" size=".004 .002 .001" contype="0" conaffinity="0" rgba="{g}"/>
            <body name="tool" pos="{x+.003} 0 {top+hh+.002}"><joint name="tool_x" type="slide" axis="1 0 0"/>
            <joint name="tool_z" type="slide" axis="0 0 1" damping="1"/>
            <geom name="actor_head" type="box" size="{.012*s} .014 {hh}" mass=".08" rgba="{a}"/>
            <geom name="actor_shaft" type="capsule" fromto="-.06 0 .004 -.01 0 .004" size=".004" mass=".015" rgba=".52 .32 .16 1"/></body>'''
            actuator='<position joint="tool_x" kp="1000" kv="25"/>'
            self.initial_actor=np.array([x+.003,top+hh+.002]);self.lift_goal=.018
        else:
            px=.010*s; pz=.020*s; inner=px+.0012; top=.05; x=off
            walls=[]
            for sign in (-1,1):
                walls.append(f'<geom name="goal_side{sign}" type="box" pos="{x+sign*(inner+.010)} 0 .027" size=".010 .03 .023" contype="2" conaffinity="1" rgba="{g}"/>')
                walls.append(f'<geom name="goal_end{sign}" type="box" pos="{x} {sign*.019} .027" size="{inner} .009 .023" contype="2" conaffinity="1" rgba="{g}"/>')
            # Opposing compliant liners create real normal contact and friction.
            for sign in (-1,1):
                walls.append(f'''<body name="liner{sign}" pos="{x+sign*(px+.0033)} 0 .029">
                <joint name="liner_j{sign}" type="slide" axis="{sign} 0 0" stiffness="1200" springref="-.001" damping="2" limited="true" range="-.003 .003"/>
                <geom name="liner_g{sign}" type="capsule" fromto="0 0 -.012 0 0 .012" size=".003" contype="2" conaffinity="1" mass=".02" friction="{cfg['friction']} .01 .001" priority="1" rgba=".47 .3 .17 1"/></body>''')
            world=f'''{''.join(walls)}<geom name="goal_floor" type="box" pos="{x} 0 .003" size=".045 .03 .003" rgba="{g}"/>
            <body name="tool" pos="{x+.003} 0 {top+pz+.002}"><joint name="tool_x" type="slide" axis="1 0 0"/>
            <joint name="tool_z" type="slide" axis="0 0 1" damping="1"/>
            <geom name="object_peg" type="box" size="{px} .009 {pz}" contype="1" conaffinity="2" mass=".08" rgba="{b}"/>
            <geom name="actor_mount" type="box" pos="0 0 {pz+.005}" size=".006 .006 .005" mass=".02" rgba="{a}"/></body>'''
            actuator='<position joint="tool_x" kp="1000" kv="25"/>'
            self.initial_actor=np.array([x+.003,top+pz+.002]);self.lift_goal=.026
        self.xml=common+world+'</worldbody><actuator>'+actuator+'</actuator></mujoco>'
        self.model=mujoco.MjModel.from_xml_string(self.xml);self.data=mujoco.MjData(self.model)
        mujoco.mj_forward(self.model,self.data)
        self.tool=int(self.model.body('tool').id)
        self.object=int(self.model.body('object').id) if self.task!='insert' else self.tool
        self.xj=int(self.model.joint('tool_x').qposadr[0]);self.zj=int(self.model.joint('tool_z').qposadr[0])
        self.renderer=mujoco.Renderer(self.model,height=resolution,width=resolution) if render else None
        self.role_ids=[]
        patterns=[('actor_',),('object_',),('part_handle',)] if self.task=='cup' else [('actor_head',),('object_',),('goal_',)] if self.task=='hammer' else [('actor_',),('object_',),('goal_',)]
        for pats in patterns:
            self.role_ids.append([i for i in range(self.model.ngeom) if (mujoco.mj_id2name(self.model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith(pats)])
        self.initial_object=self.data.xpos[self.object].copy();self.history=[]

    def public(self):
        self.renderer.update_scene(self.data,camera='side');self.renderer.disable_segmentation_rendering()
        rgb=self.renderer.render().copy();self.renderer.enable_segmentation_rendering()
        seg=self.renderer.render().copy();self.renderer.disable_segmentation_rendering()
        valid=seg[...,1]==int(self.mj.mjtObj.mjOBJ_GEOM)
        masks=np.stack([valid&np.isin(seg[...,0],ids) for ids in self.role_ids])
        boxes=[];pos=[];sizes=[];visible=[]
        scale=2*.45*np.tan(np.deg2rad(17.5))/self.resolution
        # Declared surface planes use fixed fixture/pad thickness, not object poses.
        planes = (-.015, -.018, -.005) if self.task == 'cup' else (-.014, -.015, -.002) if self.task == 'hammer' else (-.006, -.009, -.028)
        for role, mask in enumerate(masks):
            scale=2*(.45+planes[role])*np.tan(np.deg2rad(17.5))/self.resolution
            ys,xs=np.where(mask);ok=len(xs)>=4;visible.append(ok)
            if ok:
                b=np.array([xs.min(),ys.min(),xs.max()+1,ys.max()+1]);cx=(b[0]+b[2])/2;cy=(b[1]+b[3])/2
                pos.append([.025+(cx-self.resolution/2)*scale,.075-(cy-self.resolution/2)*scale]);sizes.append([(b[2]-b[0])*scale,(b[3]-b[1])*scale]);boxes.append(b)
            else:pos.append([np.nan,np.nan]);sizes.append([np.nan,np.nan]);boxes.append(np.zeros(4))
        p=self.data.xpos[self.tool][[0,2]].copy();vel=self.data.qvel[self.model.jnt_dofadr[[self.model.joint('tool_x').id,self.model.joint('tool_z').id]]].copy()
        return dict(positions=np.array(pos),sizes=np.array(sizes),visible=np.array(visible),boxes=np.array(boxes),proprio=np.r_[p,vel]),rgb,masks

    def step(self,action):
        # Cup action: world x/z gripper target and per-jaw force. Others: x target/down force.
        a=np.asarray(action);self.data.ctrl[0]=float(a[0]-self.initial_actor[0])
        if self.task=='cup':
            self.data.ctrl[1]=float(a[1]-self.initial_actor[1]);self.data.ctrl[2:4]=np.clip(a[2],0,12)
        else:
            # Exact gravity cancellation is tool proprioception, not object information.
            mass=sum(self.model.body_mass[i] for i in range(self.model.nbody) if i==self.tool)
            self.data.xfrc_applied[self.tool,2]=mass*9.81-float(np.clip(a[1],-2,5))
            if self.task=='hammer':self.data.xfrc_applied[self.object,2]=self.model.body_mass[self.object]*9.81
        for _ in range(40):self.mj.mj_step(self.model,self.data)
        self.time+=1

    def state(self):
        return dict(qpos=self.data.qpos.copy(),qvel=self.data.qvel.copy(),object=self.data.xpos[self.object].copy(),tool=self.data.xpos[self.tool].copy())

    def evaluate(self):
        s=self.state();pairs={frozenset((int(c.geom1),int(c.geom2))) for c in self.data.contact if c.dist<.001}
        if self.task=='cup':
            handle=int(self.model.geom('part_handle').id)
            contact=[frozenset((int(self.model.geom(name).id),handle)) in pairs for name in ('actor_left','actor_right')]
            lift=s['object'][2]-self.initial_object[2]
            return dict(success=bool(all(contact) and lift>.028),progress=float(lift),contact=contact)
        if self.task=='hammer':
            progress=self.initial_object[2]-s['object'][2]
            return dict(success=bool(.015<progress<.025),progress=float(progress))
        progress=self.initial_object[2]-s['object'][2]
        return dict(success=bool(progress>.025 and abs(s['object'][0]-self.cfg['offset'])<.0018),progress=float(progress))

    def close(self):
        if self.renderer is not None:self.renderer.close()


class EngineeringControl:
    """Observation-only nominal controller; force supplied for engineering sweep."""
    def __init__(self,task,force):self.task=task;self.force=force;self.initial=None;self.zgoal=None

    def __call__(self,o,t):
        pos=o['positions'];p=o['proprio'][:2]
        if self.initial is None:self.initial=pos.copy();self.zgoal=p[1];self.initial_proprio=p.copy()
        if self.task=='cup':
            # Align to visible handle; follow only while lift is maintained.
            hx,hz=pos[2] if o['visible'][2] else self.initial[2]
            if t<8:self.zgoal=hz
            elif t<75:
                followed=(hz-self.initial[2,1]) >= (p[1]-self.initial[2,1])-.006
                self.zgoal=min(self.initial[2,1]+.038,self.zgoal+.0007) if followed else hz+.002
            return np.array([hx,self.zgoal,self.force])
        if self.task=='hammer':
            x=pos[1,0];progress=self.initial[1,1]-pos[1,1]
            f=self.force if progress<.018 else 0
            return np.array([x,f])
        x=pos[2,0];progress=self.initial_proprio[1]-p[1]
        f=self.force if progress<.028 else 0
        if abs(p[0]-x)>.0015:f=0
        return np.array([x,f])
