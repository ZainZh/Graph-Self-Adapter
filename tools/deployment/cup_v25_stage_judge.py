"""V25 explicit current Codex Root fallback; model and Root evidence stay separate."""
import json
from pathlib import Path
from prove.xtrainer.agent import RealRootJudge, REPLY_SCHEMA, APPROVAL_SCHEMA, digest, write
from prove.xtrainer.io import SafetyError
from cup_v25_compact_judge import OpenRouterJudge, compact_context, MODEL

MODEL_IDENTITY = 'openrouter/' + MODEL
ROOT_IDENTITY = 'current Codex Root'
PROTOCOL_VERSION = 'cup_v25_compact_grasp_with_explicit_root_fallback'

def validate_config(config):
    if config.get('backend') != 'OpenRouter' or config.get('model') != MODEL or config.get('maximum_calls') != 6 or not config.get('key_file'):
        raise SafetyError('declared model/key and six-call semantic budget required')

class GeminiStageJudge(RealRootJudge):
    def __init__(self, root, *, task, graph_sha, identity=None, key_file, maximum_calls=6, client=None, **kwargs):
        if maximum_calls != 6: raise SafetyError('six semantic calls required')
        kwargs.setdefault('wall_timeout_sec', 900.)
        kwargs.setdefault('max_reply_age_sec', 900.)
        super().__init__(root, task=task, graph_sha=graph_sha, identity=MODEL_IDENTITY, **kwargs)
        self.client = client or OpenRouterJudge(key_file, maximum_calls=maximum_calls, clock=self.clock)
        self.api_records = []

    def ask(self, stage, facts, *, before, recapture, heartbeat=None):
        if stage.get('phase') not in ('Grasp', 'Complete'):
            raise SafetyError('unsupported semantic phase')
        if len(self.api_records) >= 6: raise SafetyError('six-call semantic budget exhausted')
        processed = set(); selected = {}; folder = None
        def bridge():
            nonlocal folder
            if heartbeat: heartbeat()
            status = json.loads((self.root / 'STATUS.json').read_text())
            path = Path(status['request']).resolve()
            if not path.is_relative_to(self.root.resolve()): raise SafetyError('request outside handoff root')
            if str(path) in processed: return
            processed.add(str(path)); folder = path.parent
            request = json.loads(path.read_text()); compact = compact_context(path)
            write(folder / 'COMPACT_CONTEXT.json', compact.context)
            write(self.root / 'STATUS.json', {'state':'awaiting_model_api', 'request':str(path), 'identity':MODEL_IDENTITY})
            def record_raw(payload):
                if payload.get('secret_present_after_redaction') is not False: raise SafetyError('redaction failed')
                write(folder / 'RAW_MODEL_RESPONSE.json', payload)
            self.client.response_recorder = record_raw
            error = None; reply = None
            try:
                reply, metrics = self.client.ask(compact)
                write(folder / 'GEMINI_PARSED_REPLY.json', reply)
            except Exception as exc:
                error = str(exc)
                metrics = getattr(exc, 'metrics', {'error':error, 'call_count':self.client.calls})
            record = {'identity':MODEL_IDENTITY, 'protocol_version':PROTOCOL_VERSION, 'metrics':metrics,
                      'API_failed':error is not None, 'model_unknown':reply is not None and reply['decision']=='unknown'}
            self.api_records.append(record); write(folder / 'API_CALL.json', record); write(self.root / 'API_CALLS.json', self.api_records)
            binding = {key:request[key] for key in ('request_id','task','stage_id','graph_sha256')}
            binding['request_sha256'] = digest(path)
            if error is not None or reply['decision'] == 'unknown':
                self.identity = ROOT_IDENTITY
                template = {**binding, 'schema':REPLY_SCHEMA, 'decision':'unknown', 'judged_facts':{f:None for f in facts},
                            'rationale':'REPLACE_WITH_ACTUAL_CURRENT_ROOT_EVIDENCE_JUDGEMENT', 'evidence_frame_ids':['f0','f2']}
                write(folder / 'ROOT_REPLY_TEMPLATE.json', template)
                handoff = {'schema':'graph-self-adapter.actual-root-fallback-handoff.v1',
                           'actual_root_identity_required':ROOT_IDENTITY, 'request':str(path),
                           'RGB_paths':[str(folder / f['path']) for f in request['frames']],
                           'compact_context':str(folder / 'COMPACT_CONTEXT.json'), 'Gemini_error':error,
                           'Gemini_reply':reply, 'requested_facts':list(facts),
                           'reply_path':str(folder / 'ROOT_REPLY.json'), 'approval_path':str(folder / 'ROOT_APPROVAL.json'),
                           'reply_template_path':str(folder / 'ROOT_REPLY_TEMPLATE.json'),
                           'approval_template':{'schema':APPROVAL_SCHEMA, 'request_id':request['request_id'],
                                                'request_sha256':binding['request_sha256'],
                                                'reply_sha256':'SHA256_OF_ACTUAL_ROOT_REPLY_BYTES', 'actual_root_identity':ROOT_IDENTITY},
                           'holding_scope':'tentative supported grasp only; not attachment proof',
                           'unknown_action':'remain unknown and stop; controlled Lift needs separately frozen graph contract',
                           'public_proprioception':request['public_robot'], 'stage_contract':request['stage_contract']}
                write(folder / 'ROOT_FALLBACK_HANDOFF.json', handoff)
                write(self.root / 'STATUS.json', {'state':'awaiting_actual_root_fallback', 'request':str(path),
                                                'handoff':str(folder / 'ROOT_FALLBACK_HANDOFF.json'), 'RGB':handoff['RGB_paths']})
                print(json.dumps({'actual_root_fallback':handoff}), flush=True)
                selected.update(semantic_source='actual current Codex Root explicit current-image judgement', Root_fallback=True)
                return
            self.identity = MODEL_IDENTITY
            legacy = {**binding, 'schema':REPLY_SCHEMA, 'decision':reply['decision'], 'judged_facts':reply['facts'],
                      'rationale':reply['reason'], 'evidence_frame_ids':reply['evidence_frame_ids']}
            write(folder / 'ROOT_REPLY.json', legacy)
            write(folder / 'ROOT_APPROVAL.json', {'schema':APPROVAL_SCHEMA, 'request_id':request['request_id'],
                'request_sha256':binding['request_sha256'], 'reply_sha256':digest(folder / 'ROOT_REPLY.json'), 'actual_root_identity':MODEL_IDENTITY})
            write(folder / 'API_ADAPTER_RECEIPT.json', {'identity':MODEL_IDENTITY, 'not_human_Root_approval':True})
            selected.update(semantic_source='Gemini visual inference', Root_fallback=False)
        previous = len(self.records)
        try: return super().ask(stage, facts, before=before, recapture=recapture, heartbeat=bridge)
        finally:
            if len(self.records) > previous:
                self.records[-1].update(selected, postreply_fresh_guard_unchanged=True)
                write(folder / 'CONSUMED.json', self.records[-1])
