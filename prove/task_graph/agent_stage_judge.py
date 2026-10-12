"""File handoff to the current Root Agent; no model service or action interface."""
from __future__ import annotations
from dataclasses import dataclass,field
import hashlib
import json
import math
from pathlib import Path

REQUEST_SCHEMA='graph-self-adapter.agent-stage-request.v1'
REPLY_SCHEMA='graph-self-adapter.agent-stage-reply.v1'
DECISIONS={'continue','complete','local_failure','structural_failure','unknown'}
REQUEST_KEYS={'schema','request_id','task','stage_id','graph_sha256','observed_physical_timestamp_sec',
 'expires_physical_timestamp_sec','stage_contract','frames','public_proprio','requested_fact_ids','input_provenance'}
CONTRACT_KEYS={'skill','actor','object','target','goal','termination'}
FRAME_KEYS={'frame_id','path','sha256','physical_timestamp_sec','view'}
PROPRIO_KEYS={'robot_timestamp_sec','q','qdot','tcp_position_m','tcp_rotation','finger_encoders',
 'actual_aperture_m','commanded_aperture_m','measurement_source','calibration_source'}
REPLY_KEYS={'schema','request_id','request_sha256','task','stage_id','graph_sha256',
 'observed_physical_timestamp_sec','decision','judged_facts','rationale','evidence_frame_ids'}
FORBIDDEN={'body_pose','world_pose','contact','contacts','force','forces','evaluator','success','gt_labels','qpos'}


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _keys(value,keys):
    if not isinstance(value,dict) or set(value)!=keys:raise ValueError('missing or non-whitelisted fields')

def _number(value):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('invalid numeric sensor timestamp/value')

def _numeric_tree(value):
    if isinstance(value,list):
        for item in value:_numeric_tree(item)
    else:_number(value)

def validate_request(request,*,now_sec,task,stage_id,graph_sha256,frame_root):
    _keys(request,REQUEST_KEYS)
    if request['schema']!=REQUEST_SCHEMA:raise ValueError('request schema mismatch')
    if not isinstance(request['request_id'],str) or not request['request_id']:raise ValueError('missing request identity')
    if (request['task'],request['stage_id'],request['graph_sha256'])!=(task,stage_id,graph_sha256):raise ValueError('task/stage/graph mismatch')
    if len(graph_sha256)!=64 or any(c not in '0123456789abcdef' for c in graph_sha256):raise ValueError('invalid graph hash')
    observed=request['observed_physical_timestamp_sec'];expiry=request['expires_physical_timestamp_sec']
    for v in (now_sec,observed,expiry):_number(v)
    if not observed<=now_sec<=expiry or expiry<=observed:raise ValueError('expired or future request')
    _keys(request['stage_contract'],CONTRACT_KEYS)
    if not all(isinstance(v,str) and v for v in request['stage_contract'].values()):raise ValueError('missing stage contract')
    ids=request['requested_fact_ids']
    if not isinstance(ids,list) or len(set(ids))!=len(ids) or any(not isinstance(x,str) or any(word in x.lower() for word in FORBIDDEN) for x in ids):raise ValueError('nonpublic fact ids')
    frames=request['frames'];seen=set();last=-math.inf
    if not isinstance(frames,list) or not 1<=len(frames)<=6:raise ValueError('one to six public RGB frames required')
    for frame in frames:
        _keys(frame,FRAME_KEYS);t=frame['physical_timestamp_sec'];_number(t)
        if not 0<=t<=observed or t<last or frame['frame_id'] in seen:raise ValueError('invalid causal frame sequence')
        seen.add(frame['frame_id']);last=t
        path=(Path(frame_root)/frame['path']).resolve()
        if not path.is_relative_to(Path(frame_root).resolve()) or path.suffix.lower() not in ('.png','.jpg','.jpeg') or digest(path)!=frame['sha256']:raise ValueError('frame path/hash mismatch')
    if observed-last>.25:raise ValueError('stale latest RGB evidence')
    if not isinstance(request['public_proprio'],dict):raise ValueError('proprio must be object')
    for arm,values in request['public_proprio'].items():
        if arm not in ('left','right'):raise ValueError('unknown public robot arm')
        _keys(values,PROPRIO_KEYS)
        if values['measurement_source'] not in ('measured_encoder','simulator_actual_joint_encoder'):raise ValueError('actual feedback missing or placeholder')
        _number(values['robot_timestamp_sec'])
        if not 0<=observed-values['robot_timestamp_sec']<=.25:raise ValueError('stale actual robot feedback')
        for k in PROPRIO_KEYS-{'measurement_source','calibration_source'}:_numeric_tree(values[k])
        for key,length in (('q',6),('qdot',6),('tcp_position_m',3),('finger_encoders',2)):
            if not isinstance(values[key],list) or len(values[key])!=length or any(isinstance(v,list) for v in values[key]):raise ValueError('malformed robot shape')
        rotation=values['tcp_rotation']
        if not isinstance(rotation,list) or len(rotation)!=3 or any(not isinstance(row,list) or len(row)!=3 for row in rotation):raise ValueError('malformed FK rotation')
        if not isinstance(values['calibration_source'],str) or not values['calibration_source']:raise ValueError('missing actual calibration provenance')
    if request['input_provenance']!='RGB_only_plus_optional_public_robot_feedback; simulator_proxy_not_hardware':raise ValueError('undeclared input provenance')
    return request


@dataclass
class AgentStageJudge:
    """Caller supplies trusted Root approval hash; replies never approve themselves."""
    consumed:set=field(default_factory=set)

    def consume(self,request_path,reply_path,*,task,stage_id,graph_sha256,now_sec,approved_reply_sha256=None):
        unknown=lambda reason:{'decision':'unknown','judged_facts':{},'reason':reason,'actions':None}
        try:
            request_path,reply_path=Path(request_path),Path(reply_path)
            request=json.loads(request_path.read_text())
            validate_request(request,now_sec=now_sec,task=task,stage_id=stage_id,graph_sha256=graph_sha256,frame_root=request_path.parent)
            if approved_reply_sha256 is None or digest(reply_path)!=approved_reply_sha256:return unknown('Root approval absent or reply hash mismatch')
            if request['request_id'] in self.consumed:return unknown('duplicate consumed request')
            reply=json.loads(reply_path.read_text());_keys(reply,REPLY_KEYS)
            if (reply['schema']!=REPLY_SCHEMA or reply['request_sha256']!=digest(request_path)
                or any(reply[k]!=request[k] for k in ('request_id','task','stage_id','graph_sha256','observed_physical_timestamp_sec'))):return unknown('reply binding mismatch')
            if reply['decision'] not in DECISIONS:return unknown('invalid decision')
            facts=reply['judged_facts']
            if not isinstance(facts,dict) or not set(facts)<=set(request['requested_fact_ids']) or any(type(v) is not bool and v is not None for v in facts.values()):return unknown('noncontracted or non-tristate facts')
            evidence=reply['evidence_frame_ids']
            if not isinstance(evidence,list) or not set(evidence)<={f['frame_id'] for f in request['frames']} or not isinstance(reply['rationale'],str):return unknown('invalid rationale/frame evidence')
            if reply['decision']!='unknown' and (not evidence or not reply['rationale'].strip()):return unknown('decision lacks visible evidence')
            self.consumed.add(request['request_id'])
            return {'decision':reply['decision'],'judged_facts':facts,'reason':reply['rationale'],'evidence_frame_ids':evidence,'actions':None}
        except (OSError,ValueError,TypeError,KeyError):return unknown('missing malformed stale or corrupt handoff')
