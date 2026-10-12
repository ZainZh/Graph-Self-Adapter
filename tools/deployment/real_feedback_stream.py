"""Bounded deterministic TCP framing over the original actual-feedback decoder.

Header/magic checks never replace SDK decode validation. No connection on import
or construction; only the existing explicit FeedbackStream.connect opens TCP.
"""
import struct
import time
from prove.xtrainer.io import FeedbackStream,decode_feedback,SafetyError

PACKET_SIZE=1440
MAGIC_OFFSET=48
MAGIC=0x123456789abcdef


class FeedbackFramer:
    def __init__(self,*,max_discard_bytes=8192,max_gap_events=8):
        if not 0<max_discard_bytes<=8192 or not 0<max_gap_events<=8:
            raise SafetyError('feedback resynchronization budget exceeds reviewed bounds')
        self.buffer=bytearray();self.header=struct.pack('<q',PACKET_SIZE);self.magic=struct.pack('<Q',MAGIC)
        self.max_discard_bytes=max_discard_bytes;self.max_gap_events=max_gap_events
        self.discarded_bytes=0;self.gap_events=0;self._in_gap=False;self.frames=0
    def _discard(self,count):
        if not count:return
        if not self._in_gap:self.gap_events+=1;self._in_gap=True
        self.discarded_bytes+=count
        if self.discarded_bytes>self.max_discard_bytes or self.gap_events>self.max_gap_events:
            raise SafetyError('feedback resynchronization budget exhausted')
        del self.buffer[:count]
    def feed(self,chunk):
        if not chunk:raise SafetyError('feedback disconnected')
        if len(chunk)>65536:raise SafetyError('feedback chunk exceeds bounded input')
        self.buffer.extend(chunk);frames=[]
        while self.buffer:
            start=self.buffer.find(self.header)
            if start<0:
                self._discard(max(0,len(self.buffer)-len(self.header)+1));break
            if start:self._discard(start)
            if len(self.buffer)<MAGIC_OFFSET+8:break
            if self.buffer[MAGIC_OFFSET:MAGIC_OFFSET+8]!=self.magic:
                self._discard(1);continue
            if len(self.buffer)<PACKET_SIZE:break
            frames.append(bytes(self.buffer[:PACKET_SIZE]));del self.buffer[:PACKET_SIZE]
            self.frames+=1;self._in_gap=False
        return frames
    def diagnostics(self):
        return {'frames':self.frames,'discarded_bytes':self.discarded_bytes,'gap_events':self.gap_events,
                'buffered_bytes':len(self.buffer),'max_discard_bytes':self.max_discard_bytes,'max_gap_events':self.max_gap_events}


class FramedFeedbackStream(FeedbackStream):
    def __init__(self,*args,sleep=time.sleep,**kwargs):
        super().__init__(*args,**kwargs)
        self._readiness_sleep=sleep;self._first_ready=False;self._readiness_wait_sec=0.
        if (self.dtype.itemsize!=PACKET_SIZE or self.dtype.fields['len'][1]!=0
            or self.dtype.fields['test_value'][1]!=MAGIC_OFFSET):
            raise SafetyError('audited SDK feedback layout mismatch')
        self.framer=FeedbackFramer()
    def _drain(self):
        try:
            previous=-1
            while not self.stop_event.is_set():
                chunk=self.sock.recv(4096)
                for raw in self.framer.feed(chunk):
                    stamp,sample=decode_feedback(raw,self.dtype,self.clock())
                    if stamp>previous:self.latest=sample;previous=stamp
        except Exception as exc:
            if not self.stop_event.is_set():self.error=exc
    def snapshot(self):
        if self._first_ready:return super().snapshot()
        started=self.clock();deadline=started+2.
        while True:
            try:
                sample=super().snapshot()
            except SafetyError as exc:
                # Stream/decode failures are terminal, never hidden by readiness waiting.
                if self.error is not None or str(exc)!='actual feedback missing/stale':raise
                sample=None
            if sample is not None and sample.mode in (5,7) and sample.enabled and not sample.error:
                self._first_ready=True;self._readiness_wait_sec=self.clock()-started
                return sample
            remaining=deadline-self.clock()
            if remaining<=0:
                self._readiness_wait_sec=self.clock()-started
                raise SafetyError('startup actual robot readiness timeout (2 seconds)')
            self._readiness_sleep(min(.01,remaining))
    def framing_diagnostics(self):
        return {**self.framer.diagnostics(),'startup_readiness_succeeded':self._first_ready,
                'startup_readiness_wait_sec':self._readiness_wait_sec}
