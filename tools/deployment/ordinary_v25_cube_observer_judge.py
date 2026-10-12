"""V19 API adapter over the frozen real bridge; no current-Root impersonation.
Legacy ROOT_REPLY/ROOT_APPROVAL filenames are program receipts, not human approval.
"""
import json
from pathlib import Path
from ordinary_v25_cube_observer_agent import RealRootJudge,REPLY_SCHEMA,APPROVAL_SCHEMA,digest,write
from prove.xtrainer.io import SafetyError
from compact_stage_judge import OpenRouterJudge,MODEL
from compact_multitask_judge import compact_context

MODEL_IDENTITY='openrouter/'+MODEL
PROTOCOL_VERSION='multitask_exact_goal_v20'
COMPACT_SHA256='7697bc7fa9ea22b81e2ef7f92221383e0c23be6f105b6a21788e3456378fe173'


def validate_config(config):
    if config.get('backend')!='OpenRouter' or config.get('model')!=MODEL or (type(config.get('maximum_calls')) is not int or config['maximum_calls']<1) or not isinstance(config.get('key_file'),str) or not config['key_file']:
        raise SafetyError('V19 requires declared OpenRouter model, separate key path, and explicit task call budget')
    if digest(Path(__file__).with_name('compact_stage_judge.py'))!=COMPACT_SHA256:raise SafetyError('frozen compact v01 source identity changed')


class MultitaskSemanticJudge(RealRootJudge):
    def __init__(self,root,*,task,graph_sha,identity=None,key_file,maximum_calls,client=None,**kwargs):
        if type(maximum_calls) is not int or maximum_calls<1:raise SafetyError('positive task boundary budget required')
        self.maximum_calls=maximum_calls
        super().__init__(root,task=task,graph_sha=graph_sha,identity=MODEL_IDENTITY,**kwargs)
        self.client=client if client is not None else OpenRouterJudge(key_file,maximum_calls=maximum_calls,clock=self.clock)
        self.api_records=[]
    def ask(self,stage,facts,*,before,recapture,heartbeat=None):
        self.identity=MODEL_IDENTITY
        if stage.get('phase') not in ('Grasp','Lift','Complete'):raise SafetyError('unsupported-phase anomaly requires stop; API only at supported boundaries')
        if len(self.api_records)>=self.maximum_calls:raise SafetyError('model task boundary budget exhausted')
        processed=set();call_record={};request_folder=None
        def bridge():
            nonlocal request_folder
            if heartbeat is not None:heartbeat()
            status=json.loads((self.root/'STATUS.json').read_text());request_path=Path(status['request']).resolve()
            if not request_path.is_relative_to(self.root.resolve()):raise SafetyError('API request outside bridge root')
            if str(request_path) in processed:return
            processed.add(str(request_path));request_folder=request_path.parent;request=json.loads(request_path.read_text())
            write(self.root/'STATUS.json',{'state':'awaiting_model_api','request':str(request_path),'backend':'OpenRouter','model':MODEL,'human_Root_reply_required':False})
            compact=compact_context(request_path)
            write(request_folder/'COMPACT_CONTEXT.json',compact.context)
            try:
                reply,metrics=self.client.ask(compact)
            except Exception as error:
                metrics=getattr(error,'metrics',{'error':str(error),'call_count':getattr(self.client,'calls',None)})
                record={'backend':'OpenRouter','model':MODEL,'identity':MODEL_IDENTITY,'protocol_version':PROTOCOL_VERSION,'metrics':metrics,'failed_closed':True}
                self.api_records.append(record);write(request_folder/'API_CALL.json',record);write(self.root/'API_CALLS.json',self.api_records)
                self.identity='Codex Root';write(self.root/'STATUS.json',{'state':'awaiting_actual_Codex_Root','request':str(request_path),'reason':str(error),'identity':self.identity});return
            call_record.update(backend='OpenRouter',model=MODEL,identity=MODEL_IDENTITY,protocol_version=PROTOCOL_VERSION,metrics=metrics,human_Root_approval=False)
            self.api_records.append(dict(call_record));write(request_folder/'API_CALL.json',call_record);write(self.root/'API_CALLS.json',self.api_records)
            if heartbeat is not None:heartbeat()
            if reply['decision'] == 'unknown' or any(value is None for value in reply['facts'].values()):
                self.identity='Codex Root';write(self.root/'STATUS.json',{'state':'awaiting_actual_Codex_Root','request':str(request_path),'reason':reply['reason'],'identity':self.identity});return
            binding={key:request[key] for key in ('request_id','task','stage_id','graph_sha256')};binding['request_sha256']=digest(request_path)
            legacy={**binding,'schema':REPLY_SCHEMA,'decision':reply['decision'],'judged_facts':reply['facts'],'rationale':reply['reason'],'evidence_frame_ids':reply['evidence_frame_ids']}
            write(request_folder/'ROOT_REPLY.json',legacy)
            write(request_folder/'ROOT_APPROVAL.json',{'schema':APPROVAL_SCHEMA,'request_id':request['request_id'],'request_sha256':binding['request_sha256'],'reply_sha256':digest(request_folder/'ROOT_REPLY.json'),'actual_root_identity':MODEL_IDENTITY})
            write(request_folder/'API_ADAPTER_RECEIPT.json',{'schema':'graph-self-adapter.api-adapter-receipt.v1','backend':'OpenRouter','model':MODEL,'identity':MODEL_IDENTITY,'not_human_Root_approval':True,'legacy_files_are_programmatic_bindings_only':True,'request_sha256':binding['request_sha256'],'reply_sha256':digest(request_folder/'ROOT_REPLY.json')})
        previous_records=len(self.records)
        try:
            judged,after=super().ask(stage,facts,before=before,recapture=recapture,heartbeat=bridge)
            if any(judged.get(f) is None for f in facts):
                if self.identity=='Codex Root':raise SafetyError('actual Root UNKNOWN stops; no blind semantic retry')
                self.identity='Codex Root'
                write(self.root/'FALLBACK_REASON.json',{'reason':'Gemini or postreply public freshness unknown; new actual Root request, no automatic API retry'})
                judged,after=RealRootJudge.ask(self,stage,facts,before=after,recapture=recapture,heartbeat=heartbeat)
                if any(judged.get(f) is None for f in facts):raise SafetyError('actual Root UNKNOWN stops; no blind semantic retry')
                return judged,after
            return judged,after
        finally:
            if len(self.records)>previous_records:
                self.records[-1].update(call_record,semantic_source=('actual current Codex Root visual verdict' if self.records[-1].get('actual_root_identity')=='Codex Root' else 'model visual inference; not current Codex Root'),postreply_fresh_guard_unchanged=True)
                if request_folder is not None:write(request_folder/'CONSUMED.json',self.records[-1])
