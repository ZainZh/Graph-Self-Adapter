"""Actual current Root may propose two fresh-frame SAM prompts; validator owns acceptance."""
import json,time
from pathlib import Path
from prove.xtrainer.agent import write,digest
from prove.xtrainer.io import SafetyError

ROLES=('white_mug','red_mug','box','white_handle','table_patch')
IDENTITY='current Codex Root'

def ground_initial_frame(frame,output,key_file,*,maximum_calls,semantic_validator,evidence_root):
    frame=Path(frame);root=Path(evidence_root)/'ACTUAL_ROOT';sha=digest(frame)
    previous=Path(evidence_root)/'UNKNOWN.json'
    feedback=json.loads(previous.read_text()) if previous.exists() else None
    for attempt in (1,2):
        folder=root/f'attempt_{attempt:02d}';folder.mkdir(parents=True,exist_ok=False)
        reply_path=folder/'ROOT_GROUNDING_REPLY.json'
        template={'schema':'graph-self-adapter.current-root-grounding-reply.v1','actual_root_identity':IDENTITY,
                  'frame_sha256':sha,'decision':'propose','role_boxes':{role:['x0','y0','x1','y1'] for role in ROLES},
                  'role_points':{'white_handle':['x','y']},'reason':'actual current Root visual proposal; SAM/RGBD validation still required'}
        request={'schema':'graph-self-adapter.current-root-grounding-request.v1','RGB':str(frame.resolve()),
                 'frame_sha256':sha,'image_width':640,'image_height':360,'coordinate_contract':'actual integer image pixels; x right/y down; exclusive box right/bottom',
                 'roles':{role:('bare unoccluded planar tabletop; not SAM' if role=='table_patch' else 'small solid local outer handle tube; avoid holes/body/robot' if role=='white_handle' else 'whole actual object SAM prompt') for role in ROLES},
                 'reply_path':str(reply_path.resolve()),'reply_template':template,'validation_feedback':feedback,
                 'attempt':attempt,'maximum_actual_Root_proposals':2,'actions_authorized':False,
                 'contact_or_task_success_asserted':False}
        write(folder/'REQUEST.json',request);write(root/'STATUS.json',{'state':'awaiting_actual_root_grounding','request':str(folder/'REQUEST.json'),'RGB':str(frame.resolve()),'reply':str(reply_path.resolve())})
        print(json.dumps({'actual_root_grounding':request}),flush=True)
        started=time.monotonic()
        while not reply_path.is_file():
            if time.monotonic()-started>600:raise SafetyError('actual Root initial grounding timeout')
            time.sleep(.1)
        reply=json.loads(reply_path.read_text())
        if set(reply)!=set(template) or reply['schema']!=template['schema'] or reply['actual_root_identity']!=IDENTITY or reply['frame_sha256']!=sha or digest(frame)!=sha:
            raise SafetyError('actual Root initial grounding schema/identity/frame mismatch')
        if reply['decision']=='unknown':raise SafetyError('actual Root initial grounding remains UNKNOWN')
        if reply['decision']!='propose' or not str(reply['reason']).strip():raise SafetyError('invalid actual Root grounding decision')
        boxes=reply['role_boxes'];points=reply['role_points']
        if set(boxes)!=set(ROLES) or set(points)!=set(('white_handle',)):raise SafetyError('complete actual Root role fields required')
        for box in boxes.values():
            if not isinstance(box,list) or len(box)!=4 or any(type(value) is not int for value in box) or not 0<=box[0]<box[2]<=640 or not 0<=box[1]<box[3]<=360:
                raise SafetyError('actual Root boxes must be valid integer current-frame pixels')
        seed=points['white_handle']
        if not isinstance(seed,list) or len(seed)!=2 or any(type(value) is not int for value in seed):raise SafetyError('actual Root foreground point required')
        payload={'frame_sha256':sha,'role_boxes':boxes,'role_points':points,'actual_root_identity':IDENTITY,
                 'grounding_source':'actual current Codex Root visual pixel proposal'}
        feedback=semantic_validator(payload);write(folder/'VALIDATION.json',feedback)
        if feedback.get('valid') is True:
            write(output,payload);write(root/'ACCEPTED.json',{'proposal':attempt,'frame_sha256':sha,'root_reply_sha256':digest(reply_path),
                  'semantic_validator_unchanged':True,'actual_Root_proposal_is_not_SAM_or_contact_truth':True})
            return payload
    raise SafetyError('actual Root initial grounding invalid after two bounded proposals')
