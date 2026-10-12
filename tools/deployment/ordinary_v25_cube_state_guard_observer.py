"""Actual Codex stage observer bridge; no automatic verdict or model API call."""
from ordinary_v25_cube_state_guard_agent import RealRootJudge
from prove.xtrainer.io import SafetyError
IDENTITY='Codex stage observer /root/apple_cube_v25'
class DirectObserver(RealRootJudge):
    def __init__(self,root,*,task,graph_sha,identity=None,key_file=None,maximum_calls=None,**kwargs):
        super().__init__(root,task=task,graph_sha=graph_sha,identity=IDENTITY,**kwargs);self.api_records=[]
    def ask(self,stage,*args,**kwargs):
        if stage['phase'] not in ('Grasp','Lift','Complete'):raise SafetyError('actual observer only at declared semantic boundaries')
        result=super().ask(stage,*args,**kwargs)
        self.records[-1]['semantic_source']='actual Codex stage observer personally reads current public frames; no scripted verdict'
        return result
