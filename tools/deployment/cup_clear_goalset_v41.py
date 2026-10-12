"""Reachable empty-gripper relation solved in own joint space, wrist orientation free."""
import numpy as np
from scipy.optimize import minimize
from prove.xtrainer.io import SafetyError
from cup_clear_retreat_v40_runtime import SharedCupExecutor as BaseExecutor,SharedSAMVision,EXTRA
from prove.xtrainer.agent import write

def solve_clearance(fk,q,g,arm,cup,pads,limits,T):
    actor=0 if arm=='left' else 1;cup=np.asarray(cup);pads=np.asarray(pads);center=np.median(cup,axis=0)
    top=float(cup[:,2].max());height=float(np.ptp(cup[:,2]));radius=float(np.linalg.norm(np.ptp(cup[:,:2],axis=0))/2);cache={}
    def state(x):
        key=np.asarray(x).tobytes()
        if key not in cache:
            full=q.copy();full[actor]=x;p,R=fk(full,g,arm);p=np.asarray(p);R=np.asarray(R)
            if actor:p=(T@np.r_[p,1])[:3];R=T[:3,:3]@R
            vertices=pads@R.T+p;cache[key]=(p,R,vertices)
        return cache[key]
    p0,R0,v0=state(q[actor]);minz=float(v0[:,2].min());mind=float(np.linalg.norm(v0[:,:2]-center[:2],axis=1).min())
    def margins(x):
        p,R,v=state(x)
        result=[float(v[:,2].min()-top-.25*height),float(np.linalg.norm(v[:,:2]-center[:2],axis=1).min()-radius-.25*height)]
        for alpha in (.25,.5,.75):
            _,_,v=state((1-alpha)*q[actor]+alpha*x)
            result.extend([float(v[:,2].min()-minz),float(np.linalg.norm(v[:,:2]-center[:2],axis=1).min()-min(mind,radius))])
        return np.asarray(result)*20
    outcome=minimize(lambda x:float(np.sum((x-q[actor])**2)),q[actor],method='SLSQP',bounds=list(zip(limits.joint_min,limits.joint_max)),constraints=[{'type':'ineq','fun':margins}],options={'maxiter':160,'ftol':1e-9})
    valid=float(margins(outcome.x).min())>=-1e-5
    report={'source':'current SAM cup + own FK/CAD/joint bounds','orientation_free_for_empty_robot':True,'optimizer_success':bool(outcome.success),'message':str(outcome.message),'verified_min_margin':float(margins(outcome.x).min()),'FK_queries':len(cache),'feasible':valid,'goal_q_actor':outcome.x.tolist(),'cup_top_m':top,'cup_height_m':height,'observed_cup_horizontal_radius_m':radius,'not_release_certificate':True}
    if not valid:return None,report
    goals=[state((1-alpha)*q[actor]+alpha*outcome.x)[:2] for alpha in (.25,.5,.75,1.)]
    return goals,report

class ReachableClearanceExecutor(BaseExecutor):
    def _code_goals_and_gripper(self,stage,pose,obs):
        if stage['stage_id']!=EXTRA:return super()._code_goals_and_gripper(stage,pose,obs)
        q=np.stack([s.q for s in self.robot.samples()]);g=self.command_g.copy();g[0 if stage['arm']=='left' else 1]=1.
        goals,report=solve_clearance(self.robot.fk,q,g,stage['arm'],self.vision.cloud(stage['object']),self.pad_library['arms'][stage['arm']]['pad_vertices_TCP_local_m'],self.trajectory_limits,self.T)
        write(self.output/'EMPTY_GRIPPER_CLEARANCE_GOALSET.json',report)
        if goals is None:raise SafetyError('no reachable own-joint gripper-clear relation')
        return goals,g
SharedCupExecutor=ReachableClearanceExecutor
