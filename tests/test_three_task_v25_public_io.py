"""Public-interface parity checks are mocks and never connect a robot."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from three_task_v25_public_io import PublicCamera, PublicRobot, PublicStop, OfflineMockTransport
from prove.xtrainer.io import SafetyError

@pytest.mark.parametrize('backend',['sim','real'])
def test_common_public_schema_unknowns_and_ack(backend):
    transport=OfflineMockTransport(); camera=PublicCamera(transport,np.eye(4),source=backend); robot=PublicRobot(transport,source=backend)
    rows=robot.samples(); assert all(s.aperture is None for s in rows)
    camera.capture(); first=camera.metadata['frame_id'];camera.capture();assert camera.metadata['frame_id']!=first
    robot.command(np.zeros((2,6)),np.ones(2),[True,True]);assert robot.last_ack.accepted is True and robot.last_ack.physical_completion is False
    with pytest.raises(SafetyError):robot.command(np.zeros((2,6)),[1,2],[True,True])
    with pytest.raises(SafetyError):robot.fk(np.zeros((2,6)),[1,1],'third')
    PublicStop(transport).stop();assert transport.stopped
    with pytest.raises(SafetyError):camera.connect()

def test_invalid_depth_is_preserved_for_observer():
    transport=OfflineMockTransport();capture=transport.capture
    def invalid():
        stamp,rgb,depth,K=capture();depth[0,0]=np.nan;return stamp,rgb,depth,K
    transport.capture=invalid
    _,_,depth,_=PublicCamera(transport,np.eye(4),source='mock').capture()
    assert np.isnan(depth[0,0])

def test_native_rpc_fake_wire_packet_and_ack_is_not_completion():
    import pickle
    from prove.xtrainer.io import NativeRPC
    class Wire:
        def __init__(self):self.sent=[]
        def connect(self,endpoint):assert endpoint=='tcp://127.0.0.1:6001'
        def send(self,raw):self.sent.append(pickle.loads(raw))
        def recv(self):return pickle.dumps({'accepted':True})
        def close(self):pass
    wire=Wire();rpc=NativeRPC(socket_factory=lambda:wire);rpc.connect()
    public=PublicRobot(rpc,source='real_fake_wire')
    public.command(np.zeros((2,6)),[.2,.7],[True,False])
    packet=wire.sent[0]
    assert packet['method']=='command_joint_state'
    np.testing.assert_allclose(packet['args']['joint_state'],[0,0,0,0,0,0,.2,0,0,0,0,0,0,.7])
    assert packet['args']['flag_in']==[True,False]
    assert public.last_ack.accepted is True and public.last_ack.physical_completion is False
    rpc.close()

def test_actual_sdk_decode_fake_wire_keeps_unknown_aperture():
    import json
    from prove.xtrainer.io import sdk_dtype,decode_feedback
    repo=Path(__file__).resolve().parents[1]
    cfg=json.loads((repo/'evaluations/gemini_graph_v19/CONFIG_PROPOSED.json').read_text())['runtime']
    dtype=sdk_dtype(cfg['sdk_file']);packet=np.zeros(1,dtype=dtype)
    packet['len']=dtype.itemsize;packet['test_value']=0x123456789abcdef;packet['time_stamp']=42
    packet['q_actual']=180.;packet['qd_actual']=90.;packet['tool_vector_actual']=[100,200,300,0,0,0]
    packet['robot_mode']=5;packet['enable_status']=1
    packet['user']=0;packet['tool']=1;packet['tool[6]']=[0,0,197,0,0,0]
    timestamp,sample=decode_feedback(packet.tobytes(),dtype,123.)
    assert timestamp==42 and sample.stamp==123. and sample.aperture is None
    np.testing.assert_allclose(sample.q,np.pi);np.testing.assert_allclose(sample.qdot,np.pi/2)
    np.testing.assert_allclose(sample.tcp,[.1,.2,.3]);np.testing.assert_allclose(sample.rotation,np.eye(3))
    assert sample.enabled and sample.mode==5 and not sample.error
