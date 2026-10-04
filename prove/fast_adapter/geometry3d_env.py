"""P41: free SE(3) tools, fixed physics, geometry-only object shifts.

No force action, gain shift, guide joint, snapping, success-dependent weld or
runtime target pose input. Oracle/facts are engineering/evaluation interfaces.
"""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation

RES = 448

def vec(x): return ' '.join(f'{v:.9g}' for v in x)
def quat(r): return np.roll(Rotation.from_matrix(r).as_quat(), 1)
def rot(q): return Rotation.from_quat(np.roll(q, -1)).as_matrix()
def configuration(task, seed, split='id'):
    g=np.random.default_rng(seed); ood=split=='geometry'
    # All geometry comes from this evaluator-owned record; never passed to policy.
    x,y=g.uniform(-.025,.025,2)
    if task=='insert':
        half=g.uniform([.010,.007],[.014,.010]) if not ood else g.uniform([.016,.011],[.020,.014])
        clearance=g.uniform(.0015,.003,2)
        angles=g.uniform([-5,-5,-10],[5,5,10]) if not ood else g.uniform([-18,-18,-40],[18,18,40])
        return dict(task=task,seed=seed,split=split,half=half.tolist(),clearance=clearance.tolist(),
                    length=float(g.uniform(.045,.060) if not ood else g.uniform(.065,.080)),
                    center=[float(x),float(y),float(g.uniform(.085,.11))],angles=angles.tolist())
    h=float(g.uniform(.06,.08) if not ood else g.uniform(.09,.12))
    return dict(task=task,seed=seed,split=split,radius=float(g.uniform(.026,.032)),height=h,
                handle_width=float(g.uniform(.025,.035) if not ood else g.uniform(.037,.05)),
                handle_height=float(g.uniform(.027,.038) if not ood else g.uniform(.045,.06)),
                handle_z=float(h*g.uniform(.5,.65)),center=[float(x-.045),float(y),.003],
                yaw=float(g.uniform(-10,10) if not ood else g.choice([-1,1])*g.uniform(20,40)))

def camera_axes(position, target, up=np.array([0.,0.,1.])):
    back=np.asarray(position)-target;back=back/np.linalg.norm(back)
    right=np.cross(up,back);right/=np.linalg.norm(right);vertical=np.cross(back,right)
    return np.stack([right,vertical,back],axis=1)

class GeometrySkill:
    def __init__(self,cfg,render=True):
        import mujoco
        self.mj=mujoco;self.cfg=cfg;self.task=cfg['task'];self.t=0
        self.renderer=None
        if self.task=='insert':
            R=Rotation.from_euler('xyz',cfg['angles'],degrees=True).as_matrix()
            p=np.array(cfg['center']);a,b=np.array(cfg['half'])+cfg['clearance'];wall=.006;depth=.05
            self.truth_target=(p,R,np.array([a,b]))
            walls=[]
            for axis,sign in [(0,-1),(0,1),(1,-1),(1,1)]:
                pos=np.array([0.,0.,-depth/2]);pos[axis]=sign*([a,b][axis]+wall/2)
                size=np.array([a+wall,b+wall,depth/2]);size[axis]=wall/2
                # A thin top cap gives a visible planar rim mask; it is part of
                # the actual fixture, not a projected truth coordinate.
                walls.append(f'<geom name="wall{axis}{sign}" type="box" pos="{vec(pos)}" size="{vec(size)}" rgba=".70 .46 .20 1"/>')
                pos[2]=.0003;size[2]=.0003
                walls.append(f'<geom name="rim{axis}{sign}" type="box" pos="{vec(pos)}" size="{vec(size)}" rgba=".87 .65 .31 1" contype="0" conaffinity="0"/>')
            world=f'<body name="object" pos="{vec(p)}" quat="{vec(quat(R))}">{"".join(walls)}<geom name="floor" type="box" pos="0 0 -.053" size="{a+wall} {b+wall} .003" rgba=".50 .31 .12 1"/></body>'
            tool_pos=np.array([.035,-.025,.23]);tool_R=np.eye(3)
            L=cfg['length'];px,py=cfg['half']
            tool=f'<geom name="peg" type="box" pos="0 0 {-L/2}" size="{px} {py} {L/2}" mass=".06" rgba=".48 .25 .78 1"/>'
            tool+='<geom name="mount" type="box" pos="0 0 .012" size=".026 .019 .01" mass=".06" rgba=".2 .55 .8 1" contype="0" conaffinity="0"/>'
            actuator='';self.goal_ids_prefix='rim';self.peg_length=L
            self.tool_calibration={}
        else:
            Rz=Rotation.from_euler('z',cfg['yaw'],degrees=True).as_matrix();p=np.array(cfg['center'])
            radius=cfg['radius'];h=cfg['height'];w=cfg['handle_width'];hh=cfg['handle_height'];hz=cfg['handle_z']
            parts=[f'<body name="object" pos="{vec(p)}" quat="{vec(quat(Rz))}"><freejoint name="cup_free"/><inertial pos="0 0 {h/2}" mass=".06" diaginertia=".00008 .00008 .00004"/>',
                   f'<geom name="cup_base" type="cylinder" pos="0 0 .003" size="{radius} .003" rgba=".64 .3 .73 1"/>']
            for i in range(16):
                angle=i*np.pi/8
                parts.append(f'<geom name="cup_wall{i}" type="box" pos="{(radius-.002)*np.cos(angle)} {(radius-.002)*np.sin(angle)} {h/2}" size=".003 {radius*np.sin(np.pi/16)} {h/2}" euler="0 0 {angle}" rgba=".64 .3 .73 1"/>')
            for name,x,z,sx,sz in [('left',radius,hz,.003,hh/2),('right',radius+w,hz,.003,hh/2),('top',radius+w/2,hz+hh/2,w/2,.003),('bottom',radius+w/2,hz-hh/2,w/2,.003)]:
                parts.append(f'<geom name="handle_{name}" type="box" pos="{x} 0 {z}" size="{sx} .004 {sz}" rgba=".89 .65 .25 1"/>')
            parts.append('</body>');world=''.join(parts)
            target_R=Rz@Rotation.from_euler('x',90,degrees=True).as_matrix()
            target_p=p+Rz@np.array([radius+w/2,-.004,hz])
            self.truth_target=(target_p,target_R,np.array([w/2+.003,hh/2+.003]))
            tool_R=Rotation.from_euler('x',90,degrees=True).as_matrix();tool_pos=np.array([.025,-.11,.12])
            # Fingers straddle the handle plane. Open/close is position controlled
            # with fixed gains across all objects; no force parameter is exposed.
            tool='<geom name="mount" type="box" pos="-.019 0 .005" size=".004 .022 .032" mass=".06" rgba=".2 .55 .8 1" contype="0" conaffinity="0"/>'
            for sign in (-1,1):
                tool+=f'<body name="finger{sign}" pos="0 0 {sign*.025}"><joint name="jaw{sign}" type="slide" axis="0 0 {-sign}" range="0 .022" limited="true" damping="2"/><geom name="pad{sign}" type="box" size=".007 .008 .003" mass=".015" rgba=".21 .68 .86 1" condim="6" friction="2 .2 .2" priority="1"/></body>'
            actuator='<actuator>'+''.join(f'<position joint="jaw{s}" kp="500" kv="6" ctrlrange="0 .022" ctrllimited="true"/>' for s in (-1,1))+'</actuator>'
            self.goal_ids_prefix='handle_';self.tool_calibration={}
        self.start_pos=tool_pos.copy();self.start_R=tool_R.copy()
        # Tool-fixed camera extrinsics, no target tracking; camera pose is derived
        # solely from tool proprioception and this constant calibration.
        cp=np.array([.09,-.14,.035]) if self.task=='insert' else np.array([0.,-.08,.13])
        look=np.array([0.,0.,-.13]) if self.task=='insert' else np.array([0.,0.,-.08])
        cr=camera_axes(cp,look,np.array([0.,1.,0.]))
        self.camera_local=(cp,cr)
        camera=f'<camera name="wrist" pos="{vec(cp)}" xyaxes="{vec(cr[:,:2].T.flatten())}" fovy="65"/>'
        gp=np.array([.36,-.48,.38]);gr=camera_axes(gp,np.array([0.,0.,.09]));self.global_camera=(gp,gr)
        xml=f'''<mujoco model="p41_{self.task}"><compiler angle="radian"/>
        <option timestep=".001" integrator="implicitfast" iterations="80" cone="elliptic"/>
        <visual><global offwidth="896" offheight="448"/><quality shadowsize="1024" offsamples="0"/></visual>
        <default><geom condim="4" friction="1.5 .01 .001" solref=".005 1" solimp=".95 .99 .001"/></default>
        <worldbody><light pos="0 -.3 .7"/><geom name="table" type="plane" size=".4 .4 .01" rgba=".27 .32 .35 1"/>
        <camera name="global" pos="{vec(gp)}" xyaxes="{vec(gr[:,:2].T.flatten())}" fovy="38"/>
        {world}<body name="command" mocap="true" pos="{vec(tool_pos)}" quat="{vec(quat(tool_R))}"/>
        <body name="tool" pos="{vec(tool_pos)}" quat="{vec(quat(tool_R))}"><freejoint name="tool_free"/>{camera}{tool}</body></worldbody>
        <equality><weld body1="command" body2="tool" solref=".003 1" torquescale="1"/></equality>{actuator}</mujoco>'''
        self.model=mujoco.MjModel.from_xml_string(xml);self.data=mujoco.MjData(self.model)
        self.tool_id=self.model.body('tool').id;self.obj_id=self.model.body('object').id
        self.mj.mj_forward(self.model,self.data)
        for _ in range(100):self.mj.mj_step(self.model,self.data)
        self.initial_object=self.data.xpos[self.obj_id].copy()
        self.goal_ids=[i for i in range(self.model.ngeom) if (self.model.geom(i).name or '').startswith(self.goal_ids_prefix)]
        if render:self.renderer=mujoco.Renderer(self.model,RES,RES)

    def calibrate_held_geometry(self):
        """Fit held cuboid extents from its visible mask and proprioception.

        No cfg dimensions, model geom_size, object pose or depth buffer is read.
        One calibrated external RGB/part-mask view is declared for this step.
        """
        from scipy.optimize import least_squares
        self.renderer.enable_segmentation_rendering();self.renderer.update_scene(self.data,camera='global');seg=self.renderer.render().copy();self.renderer.disable_segmentation_rendering()
        mask=(seg[:,:,0]==self.model.geom('peg').id)&(seg[:,:,1]==int(self.mj.mjtObj.mjOBJ_GEOM))
        yy,xx=np.nonzero(mask)
        if len(xx)<30:raise RuntimeError('Held geometry not visible for visual calibration')
        observed=np.array([xx.min()-.5,yy.min()-.5,xx.max()+.5,yy.max()+.5])
        p,R=self.proprio();C,V=self.global_camera;f=(RES/2)/np.tan(np.radians(38/2))
        def projection(dims):
            hx,hy,L=dims
            corners=np.array([[x*hx,y*hy,z*L] for x in (-1,1) for y in (-1,1) for z in (-1,0)])
            cam=(corners@R.T+p-C)@V
            uv=np.c_[RES/2+f*cam[:,0]/(-cam[:,2]),RES/2-f*cam[:,1]/(-cam[:,2])]
            return np.r_[uv.min(0),uv.max(0)]
        fit=least_squares(lambda d:projection(d)-observed,[.012,.009,.052],bounds=([.005,.004,.025],[.035,.03,.10]),xtol=1e-10,ftol=1e-10,gtol=1e-10)
        self.tool_calibration=dict(length=float(fit.x[2]),half=fit.x[:2].copy(),pixel_residual=float(np.linalg.norm(fit.fun)),source='RGB visible-mask bbox plus known own-tool pose')
        return self.tool_calibration

    def proprio(self):
        return self.data.xpos[self.tool_id].copy(),self.data.xmat[self.tool_id].reshape(3,3).copy()

    def step(self,p,R,closed=False):
        old,oldR=self.proprio();delta=np.asarray(p)-old;norm=np.linalg.norm(delta)
        self.data.mocap_pos[0]=old+delta*min(1.,.005/max(norm,1e-9))
        dr=Rotation.from_matrix(R@oldR.T).as_rotvec();mag=np.linalg.norm(dr)
        desired=Rotation.from_rotvec(dr*min(1.,.055/max(mag,1e-9))).as_matrix()@oldR
        self.data.mocap_quat[0]=quat(desired)
        if self.task=='cup':self.data.ctrl[:]=.022 if closed else 0.
        for _ in range(40):self.mj.mj_step(self.model,self.data)
        self.t+=1

    def observe(self):
        import cv2
        self.renderer.disable_segmentation_rendering();self.renderer.update_scene(self.data,camera='wrist');rgb=self.renderer.render().copy()
        self.renderer.enable_segmentation_rendering();seg=self.renderer.render().copy();self.renderer.disable_segmentation_rendering()
        mask=(np.isin(seg[:,:,0],self.goal_ids)&(seg[:,:,1]==int(self.mj.mjtObj.mjOBJ_GEOM))).astype(np.uint8)
        # Corner estimates derive from a visible part mask, not simulator points.
        contours,hierarchy=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE);corners=None
        # An interrupted frame can look like a different quadrilateral. Admit
        # only a visibly closed opening, without consulting hidden visibility.
        closed=hierarchy is not None and any(h[3]>=0 and cv2.contourArea(c)>25 for c,h in zip(contours,hierarchy[0]))
        contours=[c for c,h in zip(contours,hierarchy[0])] if hierarchy is not None else []
        if contours and (closed or self.task=='cup'):
            hull=cv2.convexHull(np.concatenate(contours));perim=cv2.arcLength(hull,True)
            for eps in (.015,.025,.04,.06):
                poly=cv2.approxPolyDP(hull,eps*perim,True).reshape(-1,2)
                if len(poly)==4:
                    points=poly.astype(float);center=points.mean(0);points=points[np.argsort(np.arctan2(points[:,1]-center[1],points[:,0]-center[0]))]
                    start=np.argmin(points.sum(1));corners=np.roll(points,-start,axis=0);break
        p,R=self.proprio();cp,cr=self.camera_local;C=p+R@cp;V=R@cr
        f=(RES/2)/np.tan(np.radians(65/2));K=np.array([[f,0,RES/2],[0,f,RES/2],[0,0,1.]])
        return dict(corners=corners,camera_position=C,camera_rotation=V,K=K,position=p,rotation=R),rgb,mask

    def render_global(self):
        self.renderer.disable_segmentation_rendering();self.renderer.update_scene(self.data,camera='global');return self.renderer.render().copy()

    def oracle(self):
        p,R,size=self.truth_target
        if self.task=='insert':return p.copy(),R.copy(),size.copy()
        # Only engineering Oracle reads the current free object's pose.
        originalR=Rotation.from_euler('z',self.cfg['yaw'],degrees=True).as_matrix()
        nowR=self.data.xmat[self.obj_id].reshape(3,3);now=self.data.xpos[self.obj_id]
        return now+nowR@originalR.T@(p-np.array(self.cfg['center'])),nowR@originalR.T@R,size.copy()

    def facts(self):
        p,R=self.proprio()
        if self.task=='insert':
            c,Q,size=self.truth_target;tip=p-R[:,2]*self.cfg['length'];relative=Q.T@(tip-c)
            angle=np.linalg.norm(Rotation.from_matrix(Q.T@R).as_rotvec())
            # Evaluation uses the physical peg pose, not its command target.
            return dict(success=bool(relative[2]<-.034 and np.max(np.abs(relative[:2]))<.003 and angle<.10),depth=float(-relative[2]),lateral=float(np.linalg.norm(relative[:2])),angle=float(angle))
        pairs={frozenset((int(c.geom1),int(c.geom2))) for c in self.data.contact if c.dist<.0005}
        target=self.model.geom('handle_right').id
        bilateral=all(frozenset((self.model.geom('pad'+str(s)).id,target)) in pairs for s in (-1,1))
        lift=self.data.xpos[self.obj_id,2]-self.initial_object[2]
        return dict(success=bool(bilateral and lift>.04),lift=float(lift),bilateral=bool(bilateral))

    def close(self):
        if self.renderer:self.renderer.close()
