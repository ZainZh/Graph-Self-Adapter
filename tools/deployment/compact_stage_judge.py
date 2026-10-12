"""Minimal shared visual context and optional one-shot OpenRouter transport.
Import/context construction never reads a key or sends a request. No robot actions.
"""
import base64
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from PIL import Image

ENDPOINT='https://openrouter.ai/api/v1/chat/completions'
MODEL='google/gemini-3.8-flash'
REASONING={'effort':'low','exclude':True}
FACTS={'holding','released','lifted','support_permission','align_ready','placement_complete'}
DECISIONS=('continue','complete','unknown','local_failure','structural_failure')
SYSTEM_PROMPT='''Judge only the requested visual facts for this stage from the supplied first/latest RGB frames and compact public context. Return JSON only. The robot's world keeps moving; this judgement does not authorize an action. Grasp while the object is supported may give only tentative holding permission, not proof of attachment or permanent contact. At a Complete stage, confirmation requires the gripper visibly opened and retreated, and the object resting stably on the declared target; target accessibility must match the stated goal relation. Commands are intentions, not measurements: an open/close command cannot prove gripping, release or success. Unknown jaw measurements remain unknown. Occlusion, ambiguous contact, insufficient temporal evidence or uncertainty require null facts and decision unknown. Do not infer hidden object poses, contact, evaluator results or task success. Do not invent evidence frames or facts. Failure reports never contain recovery actions. Use only supplied evidence IDs and a short reason (at most120 characters).'''

class JudgeError(ValueError):pass

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite(value):
    if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):raise JudgeError('finite numeric public measurement required')
    return float(value)
def scalar(value):
    if value is None:return None
    if not isinstance(value,str) or len(value)>160:raise JudgeError('short declared stage field required')
    return value

@dataclass
class CompactContext:
    context: dict
    images: tuple
    local_bindings: dict


def compact_context(request_path):
    path=Path(request_path).resolve();root=path.parent;request=json.loads(path.read_text());stage=request['stage_contract']
    facts=request['requested_facts']
    if not isinstance(facts,list) or not facts or len(set(facts))!=len(facts) or not set(facts)<=FACTS:raise JudgeError('requested facts outside contract')
    arm=stage.get('arm')
    if arm not in ('left','right'):raise JudgeError('actor arm required')
    frames=request['frames']
    if len(frames)<2 or frames[0]['frame_id']!='f0' or frames[-1]['frame_id']!='f2':raise JudgeError('actual first f0 and latest f2 required')
    stamps=[finite(row['observed_monotonic_sec']) for row in frames]
    if any(b<a for a,b in zip(stamps,stamps[1:])) or len({row['frame_id'] for row in frames})!=len(frames):raise JudgeError('frame timestamps/order invalid')
    selected=(frames[0],frames[-1]);images=[];bindings=[]
    for row in selected:
        relative=Path(row['path']);image=(root/relative).resolve()
        if relative.is_absolute() or not image.is_relative_to(root) or not image.is_file() or digest(image)!=row['sha256']:raise JudgeError('frame path/hash mismatch')
        with Image.open(image) as rgb:
            if rgb.size!=(640,360) or rgb.format!='PNG' or rgb.mode!='RGB':raise JudgeError('original640x360 RGB PNG required; no resizing')
        images.append(image);bindings.append({'frame_id':row['frame_id'],'path':str(image),'sha256':row['sha256'],'observed_monotonic_sec':row['observed_monotonic_sec']})
    actor_index=0 if arm=='left' else 1;actor=request['public_robot'][actor_index]
    aperture=actor.get('actual_aperture_m');aperture=None if aperture is None else finite(aperture)
    if aperture is not None and aperture<0:raise JudgeError('negative jaw aperture')
    movement=None
    history=[row.get('public_robot') for row in selected]
    if all(isinstance(rows,list) and len(rows)==2 for rows in history):
        positions=[rows[actor_index].get('TCP_m') for rows in history]
        if all(isinstance(p,list) and len(p)==3 for p in positions):movement=math.sqrt(sum((finite(b)-finite(a))**2 for a,b in zip(*positions)))
    context={'stage_id':scalar(request['stage_id']),'phase':scalar(stage.get('phase')),'actor':scalar(stage.get('actor')),'arm':arm,'object':scalar(stage.get('object')),'part':scalar(stage.get('object_part')),'target':scalar(stage.get('target')),'goal_relation':scalar(stage.get('goal_relation')),'requested_facts':facts,'actual_actor_jaw_m':aperture,'first_last_public_TCP_delta_norm_m':movement,'frames':[{'frame_id':row['frame_id'],'observed_monotonic_sec':row['observed_monotonic_sec']} for row in selected]}
    contract=scalar(stage.get('holding_contract'))
    context['holding_contract']=((contract+'; ') if contract else '')+'supported-tentative visual permission; not attachment/contact measurement' if stage.get('phase')=='Grasp' else contract
    context['judge_scope']='semantic facts only; controller checks pose/progress separately'
    if stage.get('phase')=='Grasp':
        context.pop('target');context.pop('goal_relation')
        context['query']='tentative supported holding permission only'
    elif stage.get('phase')=='Complete':
        context.pop('part');context.pop('holding_contract');context.pop('goal_relation')
        context['query_goal_relation']='released_and_stable_on_declared_target'
    declaration=actor.get('gripperOpenClose')
    if declaration is not None:
        if type(declaration) not in (str,bool,int,float) or isinstance(declaration,str) and len(declaration)>20:raise JudgeError('invalid gripperOpenClose public declaration')
        if isinstance(declaration,(int,float)) and not isinstance(declaration,bool):finite(declaration)
        context['gripperOpenClose_command_not_measurement']=declaration
    return CompactContext(context,tuple(images),{'request_sha256':digest(path),'frames':bindings,'graph_sha256':request.get('graph_sha256'),'request_id':request.get('request_id')})


def response_schema(compact):
    return {'type':'json_schema','json_schema':{'name':'stage_judgement','strict':True,'schema':{'type':'object','additionalProperties':False,'required':['decision','facts','evidence_frame_ids','reason'],'properties':{'decision':{'type':'string','enum':list(DECISIONS)},'facts':{'type':'object','additionalProperties':False,'required':list(compact.context['requested_facts']),'properties':{fact:{'type':['boolean','null']} for fact in compact.context['requested_facts']}},'evidence_frame_ids':{'type':'array','items':{'type':'string','enum':[row['frame_id'] for row in compact.context['frames']]},'maxItems':2,'uniqueItems':True},'reason':{'type':'string','maxLength':120}}}}}


def parse_reply(value,compact):
    if not isinstance(value,dict) or set(value)!={'decision','facts','evidence_frame_ids','reason'}:raise JudgeError('exact compact response keys required')
    decision=value['decision'];facts=value['facts'];ids=value['evidence_frame_ids'];reason=value['reason']
    allowed=set(compact.context['requested_facts']);supplied={row['frame_id'] for row in compact.context['frames']}
    if decision not in DECISIONS or not isinstance(facts,dict) or not set(facts)<=allowed or any(v is not None and type(v) is not bool for v in facts.values()):raise JudgeError('decision/facts invalid')
    if not isinstance(ids,list) or len(ids)>2 or len(set(ids))!=len(ids) or not set(ids)<=supplied:raise JudgeError('evidence IDs invalid')
    if not isinstance(reason,str) or not reason.strip() or len(reason)>120:raise JudgeError('short reason required')
    normalized={fact:facts.get(fact) for fact in compact.context['requested_facts']}
    if decision=='complete':
        if any(v is None for v in normalized.values()):decision='unknown'
        elif any(v is False for v in normalized.values()):decision='continue'
    if decision in ('unknown','local_failure','structural_failure'):normalized={fact:None for fact in normalized}
    if any(v is True for v in normalized.values()) and not ids:raise JudgeError('positive judgement lacks supplied evidence')
    if any(normalized.get(fact) is True for fact in ('holding','released','placement_complete')):
        stamps=[finite(row['observed_monotonic_sec']) for row in compact.context['frames']]
        if set(ids)!={'f0','f2'} or len(stamps)!=2 or stamps[-1]-stamps[0]<.19:raise JudgeError('positive grasp/release/completion needs f0+f2 spanning >=.19sec')
    return {'decision':decision,'facts':normalized,'evidence_frame_ids':ids,'reason':reason}


def redact(text,secret=''):
    text=str(text)
    if secret:text=text.replace(secret,'[REDACTED]')
    text=re.sub(r'data:image/[^\s"\']+','[IMAGE_DATA]',text)
    text=re.sub(r'(?i)bearer\s+[^\s"\']+','Bearer [REDACTED]',text)
    text=re.sub(r'sk-[A-Za-z0-9_-]+','[REDACTED]',text)
    return text[:240]


class OpenRouterJudge:
    def __init__(self,key_file,*,maximum_calls,opener=urllib.request.urlopen,clock=time.monotonic,reasoning=None):
        if type(maximum_calls) is not int or maximum_calls<1:raise JudgeError('explicit bounded call budget required')
        self.key_file=Path(key_file);self.maximum_calls=maximum_calls;self.calls=0;self.records=[];self.opener=opener;self.clock=clock;self.reasoning=dict(REASONING)
        if reasoning is not None and reasoning!=REASONING:raise JudgeError('only reviewed low/exclude reasoning is permitted')
    def ask(self,compact):
        if self.calls>=self.maximum_calls:raise JudgeError('API call budget exhausted')
        secret=self.key_file.read_text().strip()
        if not secret or len(secret)>512 or any(c.isspace() for c in secret):raise JudgeError('invalid API key file')
        # Hash-check again at dispatch; local identity and full paths never enter model context.
        content=[{'type':'text','text':json.dumps(compact.context,separators=(',',':'),allow_nan=False)}]
        for path,binding in zip(compact.images,compact.local_bindings['frames']):
            if digest(path)!=binding['sha256']:raise JudgeError('frame changed before API dispatch')
            content.append({'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()}})
        payload={'model':MODEL,'messages':[{'role':'system','content':SYSTEM_PROMPT},{'role':'user','content':content}],'max_tokens':512,'temperature':0,'response_format':response_schema(compact),'provider':{'require_parameters':True}}
        payload['reasoning']=self.reasoning # Root verified provider support; reasoning usage remains billable.
        req=urllib.request.Request(ENDPOINT,data=json.dumps(payload,allow_nan=False).encode(),headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'},method='POST')
        self.calls+=1;started=self.clock();safe_usage={}
        def record(error=None):
            return {'wall_sec':self.clock()-started,'usage':safe_usage,'cost':safe_usage.get('cost'),'call_count':self.calls,'request_sha256':compact.local_bindings['request_sha256'],'model':MODEL,'reasoning_requested':self.reasoning,'no_retry':True,'error':error}
        try:
            with self.opener(req,timeout=30) as response:
                raw=response.read(131073)
                if len(raw)>131072:raise JudgeError('API response size exceeded')
                body=json.loads(raw)
            usage=body.get('usage',{});safe_usage={}
            for key in ('prompt_tokens','completion_tokens','total_tokens','cost'):
                if key in usage:safe_usage[key]=finite(usage[key])
            details=usage.get('completion_tokens_details',{})
            if isinstance(details,dict):safe_usage['completion_tokens_details']={key:finite(details[key]) for key in ('reasoning_tokens','accepted_prediction_tokens','rejected_prediction_tokens') if key in details}
            value=json.loads(body['choices'][0]['message']['content']);reply=parse_reply(value,compact)
            reply['reason']=redact(reply['reason'],secret)
            metrics=record();self.records.append(metrics);return reply,metrics
        except Exception as error:
            if isinstance(error,urllib.error.HTTPError):message='HTTP '+str(error.code)+': '+redact(error.read(1024).decode(errors='replace'),secret)
            else:message='API/response error: '+redact(error,secret)
            failure=JudgeError(message);failure.metrics=record(message);self.records.append(failure.metrics)
            raise failure from None
