"""Real-world Root bridge: wall clock, fresh resampling, no simulation pause claim."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
from PIL import Image
import numpy as np
from prove.xtrainer.io import SafetyError

REQUEST_SCHEMA='graph-self-adapter.real-stage-request.v1'
REPLY_SCHEMA='graph-self-adapter.real-stage-reply.v1'
APPROVAL_SCHEMA='graph-self-adapter.real-root-approval.v1'
FACTS={'holding','released','lifted','support_permission','align_ready','placement_complete'}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name('.'+path.name+'.'+uuid.uuid4().hex)
    with temp.open('x') as stream:
        json.dump(value,stream,indent=2,allow_nan=False,default=lambda v:v.item() if isinstance(v,np.generic) else v.tolist() if isinstance(v,np.ndarray) else str(v));stream.flush();os.fsync(stream.fileno())
    os.replace(temp,path)

def drift_guard(before,after,*,max_age_sec,max_q_drift_rad,max_role_drift_m,now,ignored_role=None):
    bs,bo=before;as_,ao=after
    for sample in as_:
        if not 0<=now-sample.stamp<=max_age_sec:raise SafetyError('postreply actual feedback stale')
    if not 0<=now-ao.stamp<=max_age_sec:raise SafetyError('postreply actual RGB stale')
    if max(np.max(np.abs(a.q-b.q)) for a,b in zip(as_,bs))>max_q_drift_rad:
        raise SafetyError('actual robot drifted while Root reviewed')
    if max(np.linalg.norm(a.tcp-b.tcp) for a,b in zip(as_,bs))>.003:raise SafetyError('actual robot TCP drifted while judge reviewed')
    for role,point in bo.roles.items():
        if role==ignored_role:continue
        if point is not None:
            fresh=ao.roles.get(role)
            if fresh is None or np.linalg.norm(np.asarray(fresh)-point)>max_role_drift_m:
                raise SafetyError('visible role missing/drifted while Root reviewed: '+role)


class RealRootJudge:
    def __init__(self,root,*,task,graph_sha,identity,clock=time.monotonic,sleep=time.sleep,
                 wall_timeout_sec=300.,max_reply_age_sec=300.):
        self.root=Path(root);self.task=task;self.graph_sha=graph_sha;self.identity=identity
        self.clock=clock;self.sleep=sleep;self.timeout=wall_timeout_sec;self.max_reply_age=max_reply_age_sec
        self.records=[];self.consumed=set()
    def ask(self,stage,facts,*,before,recapture,heartbeat=None):
        if not facts or not set(facts)<=FACTS:raise SafetyError('noncontracted real facts')
        samples,obs=before;started=self.clock();rid=uuid.uuid4().hex
        pending=self.root/('.'+rid);folder=self.root/rid;pending.mkdir(parents=True,exist_ok=False)
        history=obs.history or [(obs.stamp,obs.rgb,[]) ]
        frames=[]
        for i,(stamp,rgb,public_robot) in enumerate(history[-3:]):
            name=f'RGB_{i}.png';Image.fromarray(rgb).save(pending/name)
            frames.append({'frame_id':f'f{i}','path':name,'sha256':digest(pending/name),
                           'observed_monotonic_sec':stamp,'public_robot':public_robot})
        request={'schema':REQUEST_SCHEMA,'request_id':rid,'task':self.task,'stage_id':stage['stage_id'],
            'graph_sha256':self.graph_sha,'observed_monotonic_sec':obs.stamp,'issued_monotonic_sec':started,
            'stage_contract':stage,'requested_facts':list(facts),
            'frames':frames,
            'public_robot':[{'q':s.q.tolist(),'qdot':s.qdot.tolist(),'TCP_m':s.tcp.tolist(),
                'rotation':s.rotation.tolist(),'actual_aperture_m':s.aperture,'source':s.source,'stamp':s.stamp} for s in samples],
            'clock_contract':'real_world_keeps_moving; postreply_resample_and_drift_guard_required'}
        write(pending/'REQUEST.json',request);os.replace(pending,folder)
        write(self.root/'STATUS.json',{'state':'awaiting_actual_root','request':str(folder/'REQUEST.json'),'RGB':str(folder/frames[-1]['path'])})
        while not ((folder/'ROOT_REPLY.json').exists() and (folder/'ROOT_APPROVAL.json').exists()):
            if heartbeat is not None:heartbeat()
            if self.clock()-started>=self.timeout:raise SafetyError('real Root response timeout')
            self.sleep(.1)
        reply_path=folder/'ROOT_REPLY.json';reply=json.loads(reply_path.read_text())
        approval=json.loads((folder/'ROOT_APPROVAL.json').read_text())
        bind={'request_id':rid,'request_sha256':digest(folder/'REQUEST.json'),'task':self.task,
              'stage_id':stage['stage_id'],'graph_sha256':self.graph_sha}
        expected_reply_keys=set(bind)|{'schema','decision','judged_facts','rationale','evidence_frame_ids'}
        expected_approval={'schema':APPROVAL_SCHEMA,'request_id':rid,'request_sha256':bind['request_sha256'],
                           'reply_sha256':digest(reply_path),'actual_root_identity':self.identity}
        if approval!=expected_approval or set(reply)!=expected_reply_keys or any(reply[k]!=v for k,v in bind.items()):
            raise SafetyError('real Root reply identity/hash/graph/stage mismatch')
        if any(digest(folder/f['path'])!=f['sha256'] for f in frames):raise SafetyError('real RGB evidence file changed')
        if reply['schema']!=REPLY_SCHEMA or not isinstance(reply['evidence_frame_ids'],list) or not reply['evidence_frame_ids'] or not set(reply['evidence_frame_ids'])<={f['frame_id'] for f in frames} or rid in self.consumed:
            raise SafetyError('wrong schema/evidence or repeated real reply')
        if any(f in ('holding','released','placement_complete') for f in facts) and (len(reply['evidence_frame_ids'])<2 or frames[-1]['observed_monotonic_sec']-frames[0]['observed_monotonic_sec']<.19):
            raise SafetyError('grip/release/stability judgment lacks causal multi-frame evidence')
        if reply['decision'] not in ('continue','complete','unknown','local_failure','structural_failure') or not reply['rationale'].strip():
            raise SafetyError('invalid real Root decision')
        judged=reply['judged_facts']
        if not isinstance(judged,dict) or not set(judged)<=set(facts) or any(v is not None and type(v) is not bool for v in judged.values()):
            raise SafetyError('noncontracted real Root facts')
        if reply['decision']=='unknown':judged={fact:None for fact in facts}
        self.consumed.add(rid)
        after=recapture() # Always new real images AND independently measured robot feedback.
        reason=None
        try:
            if self.clock()-started>self.max_reply_age:raise SafetyError('real Root reply wall age expired')
            if stage['phase']=='Lift':
                witness=getattr(after[1],'holding_witness',None) or {}
                if witness.get('role')!=stage['object'] or witness.get('sparse_elevated_witness_count',0)<1 or not witness.get('unique_actor_association',False):raise SafetyError('fresh unique/elevated holding witness missing; state UNKNOWN')
            drift_guard(before,after,max_age_sec=.25,max_q_drift_rad=.015,max_role_drift_m=.01,now=self.clock(),ignored_role=(stage["object"] if stage["phase"] in ("Grasp","Lift") else None))
        except SafetyError as exc:
            reason=str(exc);judged={fact:None for fact in facts}
        record={'request_id':rid,'request_sha256':bind['request_sha256'],'reply_sha256':digest(reply_path),
            'actual_root_identity':self.identity,'wall_review_sec':self.clock()-started,
            'decision':'unknown' if reason else reply['decision'],'facts':judged,'original_reply':reply,'postreply_guard_failure':reason,
            'postreply_rgb_stamp':after[1].stamp}
        self.records.append(record);write(folder/'CONSUMED.json',record)
        if reply['decision'] in ('local_failure','structural_failure'):raise SafetyError('Root '+reply['decision'])
        return judged,after
