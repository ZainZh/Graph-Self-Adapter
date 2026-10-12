"""Synthetic TCP byte streams; no live connections or private robot results."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
from prove.xtrainer.io import sdk_dtype,SafetyError

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('_feedback_framer_test',ROOT/'tools/deployment/real_feedback_stream.py')
framing=importlib.util.module_from_spec(spec);spec.loader.exec_module(framing)
SDK=Path('/home/zheng/github/RoboTwin_bin_pick/include/xtrainer_clover/dobot_control/robots/dobot_api.py')


def packet(stamp):
    dtype=sdk_dtype(SDK);raw=np.zeros(1,dtype=dtype);p=raw[0]
    for key,value in {'len':1440,'test_value':framing.MAGIC,'time_stamp':stamp,'q_actual':[90]*6,
                      'qd_actual':[0]*6,'tool_vector_actual':[100,200,300,0,0,0],'robot_mode':5,'tool':1,
                      'enable_status':1,'tool[6]':[0,0,197,0,0,0]}.items():p[key]=value
    return raw.tobytes()


def test_arbitrary_chunks_leading_junk_and_interframe_gap_preserve_packets():
    a,b=packet(1),packet(2);stream=b'leading junk'+a+b'gap'+b;framer=framing.FeedbackFramer();frames=[]
    cuts=[1,5,37,301,2,1440,11,4096]
    offset=0
    for size in cuts:
        chunk=stream[offset:offset+size];offset+=len(chunk)
        if chunk:frames.extend(framer.feed(chunk))
    if offset<len(stream):frames.extend(framer.feed(stream[offset:]))
    assert frames==[a,b] and framer.discarded_bytes==15 and framer.gap_events==2
    assert framer.frames==2 and not framer.buffer


def test_sustained_corruption_and_stream_disconnect_fail_closed():
    framer=framing.FeedbackFramer()
    with pytest.raises(SafetyError,match='budget exhausted'):framer.feed(b'x'*9000)
    with pytest.raises(SafetyError,match='disconnected'):framing.FeedbackFramer().feed(b'')
    raw=bytearray(packet(1));raw[48:56]=b'badmagic'
    assert framing.FeedbackFramer().feed(raw)==[]


def test_original_actual_decoder_and_age_checks_are_retained_without_connect():
    clock=[1.];stream=framing.FramedFeedbackStream('not_connected',SDK,clock=lambda:clock[0],
                         socket_factory=lambda *a,**kw:pytest.fail('connection in test'))
    chunks=iter([b'junk'+packet(2)[:99],packet(2)[99:],b''])
    class Socket:
        def recv(self,n):return next(chunks)
    stream.sock=Socket();stream._drain()
    assert stream.latest.aperture is None and np.allclose(stream.latest.q,np.pi/2)
    assert np.allclose(stream.latest.tcp,[.1,.2,.3])
    assert isinstance(stream.error,SafetyError)
    with pytest.raises(SafetyError,match='failed'):stream.snapshot()
    stream.error=None;stream.snapshot();clock[0]=1.3
    with pytest.raises(SafetyError,match='stale'):stream.snapshot()


def test_gap_count_is_bounded_even_with_valid_packets_between_junk():
    framer=framing.FeedbackFramer(max_gap_events=2)
    framer.feed(b'a'+packet(1));framer.feed(b'b'+packet(2))
    with pytest.raises(SafetyError,match='budget exhausted'):framer.feed(b'c'+packet(3))


def test_observed_shape_all_zero_packet_is_discarded_before_valid_frame():
    framer=framing.FeedbackFramer();zero=bytes(1440);valid=packet(8)
    # Real failure shape: length=0, magic=0, leading 32 bytes all zero.
    assert zero[:32]==bytes(32)
    frames=framer.feed(zero+valid)
    assert frames==[valid] and framer.discarded_bytes==1440 and framer.gap_events==1
    assert framer.frames==1


def readiness_stream(mode_sequence):
    from prove.xtrainer.io import decode_feedback
    clock=[0.];waits=[];counter=[0];dtype=sdk_dtype(SDK);base_packet=packet(1)
    stream=framing.FramedFeedbackStream('not_connected',SDK,clock=lambda:clock[0],sleep=lambda dt:None,
        socket_factory=lambda *a,**kw:pytest.fail('no hardware connection'))
    def publish(mode):
        raw=np.frombuffer(base_packet,dtype=dtype).copy();raw['robot_mode']=mode;raw['time_stamp']=counter[0]+1
        counter[0]+=1;_,stream.latest=decode_feedback(raw.tobytes(),dtype,clock[0])
    def sleep(dt):
        waits.append(dt);clock[0]+=dt;publish(mode_sequence[min(len(waits)-1,len(mode_sequence)-1)])
    stream._readiness_sleep=sleep
    return stream,clock,waits,publish


def test_first_snapshot_waits_for_new_real_sample_then_runtime_disabled_is_immediate():
    from prove.xtrainer.io import SafetyLayer
    stream,clock,waits,publish=readiness_stream([0,5]);sample=stream.snapshot()
    assert sample.mode==5 and sample.enabled and not sample.error and len(waits)==2
    assert stream.framing_diagnostics()['startup_readiness_succeeded']
    sample.enabled=False # Explicit later disabled sensor fixture; wrapper must not wait or alter it.
    before=len(waits);returned=stream.snapshot();assert returned is sample and not returned.enabled and len(waits)==before
    cfg={'max_age_sec':.25,'max_velocity_rad_sec':1.,'max_tracking_rad':.05}
    with pytest.raises(SafetyError,match='mode/error'):SafetyLayer(cfg,lambda:clock[0]).samples([returned,returned])


def test_continuous_not_ready_times_out_in_two_seconds_without_mutating_mode():
    stream,clock,waits,publish=readiness_stream([0]);publish(0)
    with pytest.raises(SafetyError,match='readiness timeout'):stream.snapshot()
    assert clock[0]<=2.00000001 and stream.latest.mode==0 and not stream._first_ready


def test_startup_does_not_swallow_terminal_decode_or_stream_error():
    stream,clock,waits,publish=readiness_stream([5]);stream.error=SafetyError('feedback packet sentinel mismatch')
    with pytest.raises(SafetyError,match='sentinel'):stream.snapshot()
    assert not waits
