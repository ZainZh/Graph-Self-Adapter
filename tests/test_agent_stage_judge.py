import json
from pathlib import Path
import pytest
from prove.task_graph.agent_stage_judge import AgentStageJudge,REQUEST_SCHEMA,REPLY_SCHEMA,digest,validate_request


def pair(tmp_path):
    frame=tmp_path/'rgb.png';frame.write_bytes(b'publicRGB')
    request={'schema':REQUEST_SCHEMA,'request_id':'r1','task':'knife','stage_id':'align','graph_sha256':'a'*64,
        'observed_physical_timestamp_sec':5.,'expires_physical_timestamp_sec':5.25,
        'stage_contract':{'skill':'Align','actor':'left','object':'knife','target':'board','goal':'supported','termination':'visible placement semantics'},
        'frames':[{'frame_id':'f0','path':'rgb.png','sha256':digest(frame),'physical_timestamp_sec':5.,'view':'overview'}],
        'public_proprio':{},'requested_fact_ids':['placement_visually_stable'],
        'input_provenance':'RGB_only_plus_optional_public_robot_feedback; simulator_proxy_not_hardware'}
    rp=tmp_path/'request.json';rp.write_text(json.dumps(request))
    reply={k:request[k] for k in ('request_id','task','stage_id','graph_sha256','observed_physical_timestamp_sec')}
    reply.update(schema=REPLY_SCHEMA,request_sha256=digest(rp),decision='complete',
        judged_facts={'placement_visually_stable':True},rationale='Visible relation evidence only',evidence_frame_ids=['f0'])
    pp=tmp_path/'reply.json';pp.write_text(json.dumps(reply));return rp,pp,request,reply


def consume(j,rp,pp,**kw):
    return j.consume(rp,pp,task='knife',stage_id='align',graph_sha256='a'*64,now_sec=kw.pop('now_sec',5.1),**kw)


def test_root_approval_required_and_duplicate_unknown_without_actions(tmp_path):
    rp,pp,_,_=pair(tmp_path);j=AgentStageJudge()
    assert consume(j,rp,pp)['decision']=='unknown'
    result=consume(j,rp,pp,approved_reply_sha256=digest(pp))
    assert result['decision']=='complete' and result['actions'] is None
    assert consume(j,rp,pp,approved_reply_sha256=digest(pp))['decision']=='unknown'


@pytest.mark.parametrize('case',['expired','stage','graph','request_hash','private_request','private_reply','future_frame','corrupt_frame','unapproved','invalid_decision','nontri','bad_evidence'])
def test_bad_or_stale_handoff_unknown(tmp_path,case):
    rp,pp,request,reply=pair(tmp_path);now=5.1
    if case=='expired':now=5.3
    elif case=='stage':reply['stage_id']='lift'
    elif case=='graph':reply['graph_sha256']='b'*64
    elif case=='request_hash':reply['request_sha256']='b'*64
    elif case=='private_request':request['contacts']={'knife':True}
    elif case=='private_reply':reply['success']=True
    elif case=='future_frame':request['frames'][0]['physical_timestamp_sec']=5.2
    elif case=='corrupt_frame':(tmp_path/'rgb.png').write_bytes(b'changed')
    elif case=='invalid_decision':reply['decision']='move_robot'
    elif case=='nontri':reply['judged_facts']['placement_visually_stable']='true'
    elif case=='bad_evidence':reply['evidence_frame_ids']=['unseen']
    rp.write_text(json.dumps(request));pp.write_text(json.dumps(reply))
    approved='bad' if case=='unapproved' else digest(pp)
    assert consume(AgentStageJudge(),rp,pp,now_sec=now,approved_reply_sha256=approved)['decision']=='unknown'


def test_public_proprio_cannot_be_command_placeholder_or_stale(tmp_path):
    _,_,request,_=pair(tmp_path)
    arm={'robot_timestamp_sec':5.,'q':[0]*6,'qdot':[0]*6,'tcp_position_m':[0]*3,
        'tcp_rotation':[[1,0,0],[0,1,0],[0,0,1]],'finger_encoders':[0,0],'actual_aperture_m':.01,
        'commanded_aperture_m':0.,'measurement_source':'simulator_actual_joint_encoder','calibration_source':'proxy'}
    request['public_proprio']={'left':arm}
    kwargs=dict(now_sec=5.1,task='knife',stage_id='align',graph_sha256='a'*64,frame_root=tmp_path)
    validate_request(request,**kwargs)
    for key,value in [('measurement_source','command_placeholder'),('robot_timestamp_sec',4.),('q',[0]),('actual_aperture_m',float('nan'))]:
        bad=dict(arm);bad[key]=value;request['public_proprio']={'left':bad}
        with pytest.raises(ValueError):validate_request(request,**kwargs)


def test_offline_selection_only_numeric_times_and_public_robot_arrays():
    import numpy as np
    from tools.evidence.build_agent_stage_evidence import sample_indices,public_proprio
    scope={'earliest_time_sec':4.8,'cutoff_time_sec':5.599,'sample_times_sec':[4.8,5.1,5.599]}
    times=np.array([4.761,4.862,5.064,5.498,5.599,5.7])
    assert sample_indices(times,scope)==[1,2,4]
    class Archive(dict):
        @property
        def files(self):return list(self)
        def __getitem__(self,key):
            if 'role_' in key:raise AssertionError('hidden role geometry read')
            return super().__getitem__(key)
    a=Archive(packet00000_timestamp=np.array(5.5),packet00000_role_knife='forbidden')
    for arm in ('left','right'):
        for key,value in {'q':[0]*6,'qdot':[0]*6,'tcp_position_m':[0]*3,'tcp_rotation':np.eye(3),
            'finger_encoders':[0,0],'actual_aperture_m':.015,'commanded_aperture_m':0.,'timestamp_sec':5.5}.items():a['packet00000_'+arm+'_'+key]=np.array(value)
    values=public_proprio(a,5.599)
    assert values['left']['actual_aperture_m']==.015 and values['left']['commanded_aperture_m']==0.
    assert public_proprio(a,6.)=={}


def test_fresh_observed_timestamp_cannot_refresh_old_RGB(tmp_path):
    _,_,request,_=pair(tmp_path)
    request['frames'][0]['physical_timestamp_sec']=4.7
    with pytest.raises(ValueError,match='stale latest RGB'):
        validate_request(request,now_sec=5.1,task='knife',stage_id='align',graph_sha256='a'*64,frame_root=tmp_path)
