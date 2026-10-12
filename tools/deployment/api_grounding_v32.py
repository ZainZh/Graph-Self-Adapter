"""Automatic current-RGB grounding via OpenRouter; no actual Codex impersonation."""
import base64,hashlib,io,json,os,time,urllib.request,urllib.error
from pathlib import Path
from PIL import Image
from prove.xtrainer.io import SafetyError
MODEL='openai/gpt-6-astra'
IDENTITY='openrouter/'+MODEL
ENDPOINT='https://openrouter.ai/api/v1/chat/completions'
KEY=Path.home()/'.config/graph-self-adapter/openrouter.key'
ROLES={
 'collect_apple':{'apple':'whole red apple','bowl':'whole receiving bowl','table_patch':'bare unobstructed tabletop patch, avoid objects/robot/shadows edges'},
 'collect_cubes':{'yellow_cube':'whole yellow cube','red_cube_left':'red cube on left side of image','blue_cube':'whole blue cube','orange_cube':'whole orange cube','red_cube_right':'red cube on right side of image','bowl':'whole receiving bowl','table_patch':'bare planar tabletop without objects/robot'},
 'put_mug_on_box':{'white_mug':'whole white mug including handle','red_mug':'whole red mug','box':'whole brown wooden placement box','white_handle':'narrow solid visible outer handle tube of white mug, not rim/interior/hole','table_patch':'bare planar tabletop, not objects, robot or table edge'}}

def validate(value,roles,width,height):
    if not isinstance(value,dict) or set(value)!={'status','role_boxes','role_points','reason'}:raise SafetyError('grounding exact JSON fields missing')
    if value['status']!='proposal':raise SafetyError('API grounding UNKNOWN: '+str(value.get('reason',''))[:160])
    boxes=value['role_boxes'];points=value['role_points']
    if not isinstance(boxes,dict) or set(boxes)!=set(roles):raise SafetyError('API grounding role set mismatch')
    for name,box in boxes.items():
        if not isinstance(box,list) or len(box)!=4 or any(type(x)is not int for x in box):raise SafetyError('API box must contain integer pixels')
        if not 0<=box[0]<box[2]<=width or not 0<=box[1]<box[3]<=height:raise SafetyError('API box out of image')
    required={'white_handle'} if 'white_handle' in roles else set()
    if not isinstance(points,dict) or set(points)!=required:raise SafetyError('API foreground point fields mismatch')
    if required:
        seed=points['white_handle']
        if not isinstance(seed,list) or len(seed)!=2 or any(type(x)is not int for x in seed):raise SafetyError('API seed invalid')
        for role in ('white_mug','white_handle'):
            b=boxes[role]
            if not b[0]<=seed[0]<b[2] or not b[1]<=seed[1]<b[3]:raise SafetyError('API seed outside material prompt boxes')
    if not isinstance(value['reason'],str) or not value['reason'].strip():raise SafetyError('grounding reason required')
    return value

def proposal(frame,roles,evidence,*,feedback=None,timeout=45,opener=urllib.request.urlopen):
    frame=Path(frame);evidence=Path(evidence);evidence.mkdir(parents=True,exist_ok=False)
    raw=frame.read_bytes();sha=hashlib.sha256(raw).hexdigest()
    with Image.open(frame) as im:width,height=im.size
    context={'image_width':width,'image_height':height,'coordinate_system':'integer original image pixels [x0,y0,x1,y1], exclusive right/bottom','roles':roles,'previous_validation':feedback,
       'output':{'status':'proposal or unknown','role_boxes':{n:['x0','y0','x1','y1'] for n in roles},'role_points':{'white_handle':['x','y']} if 'white_handle' in roles else {},'reason':'short visual explanation'}}
    body={'model':MODEL,'messages':[{'role':'system','content':'You locate requested objects in the supplied CURRENT RGB. Return JSON only using exactly the requested output fields. Boxes tightly cover visible whole objects except explicit local part or tabletop ROI. For cup handle select a narrow solid outer tube; foreground point must be visible material inside both mug and local part boxes. Use ORIGINAL INTEGER PIXELS, not 0..1000 normalization. Unknown/occluded roles must return status unknown, never invent geometry. No robot actions or contact certification.'},{'role':'user','content':[{'type':'text','text':json.dumps(context)},{'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(raw).decode(),'detail':'high'}}]}],'response_format':{'type':'json_object'},'max_tokens':2000,'reasoning':{'effort':'low','exclude':True},'provider':{'allow_fallbacks':False}}
    (evidence/'REQUEST.json').write_text(json.dumps({'model':MODEL,'frame_source':str(frame.resolve()),'frame_sha256':sha,'context':context,'timeout_sec':timeout,'human_root':False},indent=2))
    key=KEY.read_text().strip()
    req=urllib.request.Request(ENDPOINT,data=json.dumps(body).encode(),headers={'Authorization':'Bearer '+key,'Content-Type':'application/json'})
    started=time.monotonic()
    try:
        with opener(req,timeout=timeout) as response:data=json.loads(response.read())
        (evidence/'RESPONSE.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
        content=data['choices'][0]['message']['content'];value=json.loads(content)
        value=validate(value,roles,width,height)
        if hashlib.sha256(frame.read_bytes()).hexdigest()!=sha:raise SafetyError('API RGB changed during request')
        payload={'frame_sha256':sha,'role_boxes':value['role_boxes'],'role_points':value['role_points'],'grounding_source':IDENTITY,'reason':value['reason']}
        (evidence/'PROPOSAL.json').write_text(json.dumps(payload,indent=2));return payload
    except urllib.error.HTTPError as error:
        (evidence/'ERROR.json').write_text(json.dumps({'error':'HTTP '+str(error.code),'model':MODEL}))
        raise SafetyError('OpenRouter grounding HTTP '+str(error.code)) from None
    finally:
        (evidence/'METRICS.json').write_text(json.dumps({'model':MODEL,'elapsed_sec':time.monotonic()-started,'human_root':False,'private_inputs':False},indent=2))

def cup_ground(frame,output,key_file,*,maximum_calls,semantic_validator,evidence_root):
    root=Path(evidence_root)/'ASTRA_API';feedback=None
    for attempt in range(1,maximum_calls+1):
        payload=proposal(frame,ROLES['put_mug_on_box'],root/f'attempt_{attempt:02d}',feedback=feedback)
        feedback=semantic_validator(payload)
        (root/f'attempt_{attempt:02d}'/'SAM_RGBD_VALIDATION.json').write_text(json.dumps(feedback,indent=2))
        if feedback.get('valid'):
            Path(output).write_text(json.dumps(payload,indent=2));return payload
    raise SafetyError('API prompts failed unchanged SAM/RGBD part validation; no manual fallback')

def install(task,entry):
    # Process-local new candidate. Historical source files and live episodes are untouched.
    import prove.xtrainer.perception as perception
    if task=='put_mug_on_box':
        import cup_v25_root_grounding_vision_v2 as v2
        import cup_v25_root_grounding_vision_v3 as v3
        v2.ground_initial_frame=cup_ground
        original=v3.RootFallbackGroundingVision._initialize_autonomously
        def initialize(self):
            self.config['initial_grounding_provider']='gemini_then_codex_root'
            obs=original(self);obs.source='RGBD+actual_SAM2+'+IDENTITY+'+functional_part_gate'
            p=Path(self.config['autonomous_grounding_evidence_root'])/'INITIAL_GROUNDING_SOURCE.json'
            p.write_text(json.dumps({'source':IDENTITY,'human_root':False,'SAM_RGBD_part_validation_unchanged':True},indent=2))
            return obs
        v3.RootFallbackGroundingVision._initialize_autonomously=initialize
        entry.entry.SOURCES+= (Path(__file__),Path(__file__).with_name('api_stage_judge_v32.py'),Path(__file__).with_name('api_v28_cup_entry.py'))
    else:
        original=perception.LiveVision.capture
        def capture(self):
            if self.tracker is None and not self.config.get('role_boxes'):
                real_camera=self.camera;sample=real_camera.capture();_,rgb,_,_=sample
                b=io.BytesIO();Image.fromarray(rgb).save(b,format='PNG');frame=Path(self.config['prompt_frame_output']);frame.parent.mkdir(parents=True,exist_ok=True);frame.write_bytes(b.getvalue())
                payload=proposal(frame,ROLES[task],frame.parent/'API_GROUNDING/attempt_01')
                self.config['prompt_frame_sha256']=payload['frame_sha256'];self.config['role_boxes']=payload['role_boxes']
                Path(self.config['prompt_reply_path']).write_text(json.dumps(payload,indent=2))
                class RetainedFirstCapture:
                    def __init__(self):self.once=True
                    def __getattr__(self,name):return getattr(real_camera,name)
                    def capture(self):
                        if self.once:self.once=False;return sample
                        return real_camera.capture()
                self.camera=RetainedFirstCapture()
                try:obs=original(self)
                finally:self.camera=real_camera
                obs.source=obs.source+'+'+IDENTITY;return obs
            return original(self)
        perception.LiveVision.capture=capture
        entry.entry.base.SOURCES+=(Path(__file__),Path(__file__).with_name('api_stage_judge_v32.py'),Path(__file__).with_name('api_v28_apple_entry.py' if task=='collect_apple' else 'api_v28_cube_entry.py'))

    backend=entry.entry.base if task=='put_mug_on_box' else entry.entry.backend
    old_main=backend.main
    def backend_main(argv=None):
        from api_stage_judge_v32 import APIStageJudge
        from argparse import ArgumentParser
        parser=ArgumentParser(add_help=False);parser.add_argument('--config',type=Path,required=True);args,_=parser.parse_known_args(argv)
        cfg=json.loads(args.config.read_text())
        def judge(*a,**kw):return APIStageJudge(*a,**kw,maximum_calls=cfg['runtime']['stage_judge']['maximum_calls'])
        backend.RealRootJudge=judge
        return old_main(argv)
    backend.main=backend_main
