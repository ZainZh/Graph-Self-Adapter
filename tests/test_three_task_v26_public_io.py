from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/deployment'))
from three_task_v26_public_io import BackendBundle,PublicRobot,PublicCamera,PreparedRealTransport

def test_real_components_have_independent_lifecycle():
    events=[]
    class Part:
        def __init__(self,name):self.name=name
        def connect(self):events.append('connect:'+self.name)
        def close(self):events.append('close:'+self.name)
    transport=PreparedRealTransport.__new__(PreparedRealTransport)
    transport.rpc=Part('RPC');transport.streams=[Part('feedbackL'),Part('feedbackR')]
    transport.stops=[Part('stopL'),Part('stopR')];transport.camera=Part('camera')
    bundle=BackendBundle(PublicRobot(transport,source='real_fake'),PublicCamera(transport.camera,np.eye(4),source='real_fake'),[], 'real',transport)
    assert events==[]
    bundle.connect();assert events.count('connect:camera')==1 and events.count('connect:RPC')==1
    bundle.close();assert events.count('close:camera')==1 and events.count('close:RPC')==1
