"""Actual current Root judges final white Complete; Gemini handles other nodes."""
import json
from pathlib import Path
from cup_v25_stage_judge import GeminiStageJudge as BaseJudge, ROOT_IDENTITY, MODEL_IDENTITY, validate_config
from cup_v25_compact_judge import compact_context
from prove.xtrainer.agent import RealRootJudge, REPLY_SCHEMA, APPROVAL_SCHEMA, digest, write

class GeminiStageJudge(BaseJudge):
    def ask(self,stage,facts,*,before,recapture,heartbeat=None):
        if stage['cycle_id']!='white_direct' or stage['phase']!='Complete':
            return super().ask(stage,facts,before=before,recapture=recapture,heartbeat=heartbeat)
        self.identity=ROOT_IDENTITY;processed=set()
        def handoff():
            if heartbeat:heartbeat()
            status=json.loads((self.root/'STATUS.json').read_text());path=Path(status['request'])
            if str(path) in processed:return
            processed.add(str(path));folder=path.parent;request=json.loads(path.read_text())
            compact=compact_context(path);write(folder/'COMPACT_CONTEXT.json',compact.context)
            binding={key:request[key] for key in ('request_id','task','stage_id','graph_sha256')};binding['request_sha256']=digest(path)
            write(folder/'ROOT_REPLY_TEMPLATE.json',{**binding,'schema':REPLY_SCHEMA,'decision':'unknown',
                  'judged_facts':{fact:None for fact in facts},'rationale':'REPLACE_BY_ACTUAL_ROOT_VISUAL_DECISION','evidence_frame_ids':['f0','f2']})
            report={'actual_root_identity_required':ROOT_IDENTITY,'request':str(path),
                    'RGB_paths':[str(folder/frame['path']) for frame in request['frames']],
                    'reply_path':str(folder/'ROOT_REPLY.json'),'approval_path':str(folder/'ROOT_APPROVAL.json'),
                    'reply_template_path':str(folder/'ROOT_REPLY_TEMPLATE.json'),
                    'approval_template':{'schema':APPROVAL_SCHEMA,'request_id':request['request_id'],
                        'request_sha256':binding['request_sha256'],'reply_sha256':'SHA256_OF_ACTUAL_ROOT_REPLY',
                        'actual_root_identity':ROOT_IDENTITY},
                    'requested_facts':list(facts),'source':'actual current Root required; no API invocation for this Complete'}
            write(folder/'ROOT_FALLBACK_HANDOFF.json',report)
            write(self.root/'STATUS.json',{'state':'awaiting_actual_root_white_complete','request':str(path),'handoff':str(folder/'ROOT_FALLBACK_HANDOFF.json'),'RGB':report['RGB_paths']})
            print(json.dumps({'actual_root_white_complete':report}),flush=True)
        previous=len(self.records)
        try:return RealRootJudge.ask(self,stage,facts,before=before,recapture=recapture,heartbeat=handoff)
        finally:
            if len(self.records)>previous:
                self.records[-1].update(semantic_source='actual current Codex Root current RGB visual final Complete',API_invoked=False)
