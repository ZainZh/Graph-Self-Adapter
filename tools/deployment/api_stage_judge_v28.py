"""Astra visual stage judge over original fresh-observation guard; never human approval."""
import base64,json,time,urllib.request
from pathlib import Path
from prove.xtrainer.agent import RealRootJudge,REPLY_SCHEMA,APPROVAL_SCHEMA,digest,write
from prove.xtrainer.io import SafetyError
from api_grounding_v28 import MODEL,IDENTITY,ENDPOINT,KEY
class APIStageJudge(RealRootJudge):
    def __init__(self,*args,maximum_calls=20,**kwargs):
        kwargs['identity']=IDENTITY;super().__init__(*args,**kwargs);self.maximum_calls=maximum_calls;self.api_calls=0
    def ask(self,stage,facts,*,before,recapture,heartbeat=None):
        processed=set()
        def respond():
            if heartbeat:heartbeat()
            status=json.loads((self.root/'STATUS.json').read_text());path=Path(status['request'])
            if str(path) in processed:return
            processed.add(str(path))
            if self.api_calls>=self.maximum_calls:raise SafetyError('Astra stage API call budget exhausted')
            self.api_calls+=1;request=json.loads(path.read_text());folder=path.parent
            # Request is produced from current/public observations by the original judge.
            from compact_stage_judge import SYSTEM_PROMPT
            if self.task=='put_mug_on_box':
                from cup_v25_compact_judge import compact_context
            else:
                from compact_multitask_judge import compact_context
            compact=compact_context(path);write(folder/'COMPACT_CONTEXT.json',compact.context)
            content=[{'type':'text','text':json.dumps(compact.context)}]
            ids=set()
            for frame in request['frames']:
                fid=frame['frame_id'];ids.add(fid);image=folder/frame['path']
                content.append({'type':'text','text':'Frame ID '+fid})
                content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(image.read_bytes()).decode(),'detail':'high'}})
            body={'model':MODEL,'messages':[{'role':'system','content':SYSTEM_PROMPT+' Return exact JSON fields decision,facts,evidence_frame_ids,reason. At Grasp judge only the declared supported-tentative permission; actual object-following confirmation belongs to Lift. Do not turn tentative permission into measured contact certification.'},{'role':'user','content':content}],'response_format':{'type':'json_object'},'max_tokens':1800,'reasoning':{'effort':'low','exclude':True},'provider':{'allow_fallbacks':False}}
            req=urllib.request.Request(ENDPOINT,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+KEY.read_text().strip(),'Content-Type':'application/json'})
            started=time.monotonic()
            with urllib.request.urlopen(req,timeout=45) as response:data=json.loads(response.read())
            write(folder/'ASTRA_API_RESPONSE.json',data);value=json.loads(data['choices'][0]['message']['content'])
            if set(value)!={'decision','facts','evidence_frame_ids','reason'} or set(value['facts'])!=set(facts):raise SafetyError('Astra judge JSON schema mismatch')
            if value['decision'] not in ('continue','complete','unknown','local_failure','structural_failure') or any(v is not None and type(v)is not bool for v in value['facts'].values()):raise SafetyError('Astra judge facts/decision invalid')
            if not isinstance(value['evidence_frame_ids'],list) or not set(value['evidence_frame_ids'])<=ids:raise SafetyError('Astra judge unsupported frame IDs')
            binding={k:request[k] for k in ('request_id','task','stage_id','graph_sha256')};binding['request_sha256']=digest(path)
            reply={**binding,'schema':REPLY_SCHEMA,'decision':value['decision'],'judged_facts':value['facts'],'rationale':value['reason'],'evidence_frame_ids':value['evidence_frame_ids']}
            write(folder/'ROOT_REPLY.json',reply)
            write(folder/'ROOT_APPROVAL.json',{'schema':APPROVAL_SCHEMA,'request_id':request['request_id'],'request_sha256':digest(path),'reply_sha256':digest(folder/'ROOT_REPLY.json'),'actual_root_identity':IDENTITY})
            write(folder/'API_ADAPTER_RECEIPT.json',{'identity':IDENTITY,'human_Root_approval':False,'legacy_filenames_are_protocol_bindings_only':True,'elapsed_sec':time.monotonic()-started,'call_index':self.api_calls})
        return super().ask(stage,facts,before=before,recapture=recapture,heartbeat=respond)
