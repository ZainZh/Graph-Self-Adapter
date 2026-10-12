"""Current actual-observer body box re-prompts existing SAM object ID, preserving others."""
import hashlib,importlib.util,json,time,uuid
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from prove.xtrainer.io import SafetyError
from prove.xtrainer.agent import write,digest
from prove.xtrainer.perception import LiveVision,role_geometry
from ordinary_v25_cube_stationary_judge_runtime import StationaryJudgeExecutor
from ordinary_v25_repair_runtime import EvidenceVision
IDENTITY='Codex stage observer /root/apple_cube_v25'
class RelocalizingVision(EvidenceVision):
    def _tracker(self):
        raw=LiveVision._tracker(self);self.raw_tracker=raw;owner=self
        class Tracker:
            def record(self,rgb,masks,method):
                index=owner.frame_count;owner.frame_count+=1;owner.public_masks=masks;owner.evidence_root.mkdir(parents=True,exist_ok=True);archive=owner.evidence_root/('RAW_MASKS_'+str(index)+'.npz');np.savez_compressed(archive,**masks)
                owner.mask_records.append({'frame_index':index,'wall_recorded_sec':owner.clock(),'physical_time_sec':getattr(owner.camera,'last_physical_time',None),'method':method,'stage_id':(owner.stage_context() or {}).get('stage_id'),'mask_source':'actual_SAM2','mask_pixels':{k:int(np.asarray(v,bool).sum()) for k,v in masks.items()},'masks':{'path':str(archive),'sha256':digest(archive)}})
            def initialize(self,rgb,boxes):
                masks=raw.initialize(rgb,boxes);self.record(rgb,masks,'actual observer initial boxes');return masks
            def track(self,rgb):
                masks=raw.track(rgb);self.record(rgb,masks,'SAM temporal tracking');return masks
        return Tracker()
    def relocalize(self,role,root,heartbeat):
        obs=self.capture();stamp,rgb,depth,K=self.camera.latest_public_capture;folder=Path(root)/uuid.uuid4().hex;folder.mkdir(parents=True);frame=folder/'RGB_CURRENT.png';Image.fromarray(rgb).save(frame);sha=digest(frame);immutable=folder/('RGB_'+sha+'.png');frame.rename(immutable);np.savez_compressed(folder/'RGBD_CURRENT.npz',depth_m=depth,K=K,T=self.T,stamp=np.asarray(stamp));request={'role':role,'frame':str(immutable),'frame_sha256':sha,'stamp':stamp,'identity':IDENTITY,'instruction':'Actual observer draw tight visible current in-hand cube body bbox, excluding fingers/background; real SAM re-prompt same existing roleID, no motion or success approval','old_mask_pixels':int(np.asarray(self.public_masks[role],bool).sum())};write(folder/'REQUEST.json',request);write(Path(root).parent/'RELOCALIZATION_STATUS.json',{'request':str(folder/'REQUEST.json'),'frame':str(immutable),'role':role,'state':'awaiting_actual_Codex_body_box'})
        started=self.clock()
        while not (folder/'REPLY.json').exists():
            heartbeat()
            if self.clock()-started>300:raise SafetyError('actual body bbox relocalization timed out')
            time.sleep(.1)
        reply=json.loads((folder/'REPLY.json').read_text());box=reply.get('box')
        if reply.get('identity')!=IDENTITY or reply.get('frame_sha256')!=sha or reply.get('role')!=role or not isinstance(box,list) or len(box)!=4 or any(type(v) is not int for v in box):raise SafetyError('actual relocalization binding/bbox invalid')
        x0,y0,x1,y1=box
        if not 0<=x0<x1<=rgb.shape[1] or not 0<=y0<y1<=rgb.shape[0]:raise SafetyError('actual body box outside image')
        path=Path(self.config['sam_utility']);spec=importlib.util.spec_from_file_location('_relocalizing_sam_utils',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);raw=self.raw_tracker
        kwargs=module.prompt_spec_to_sam2_kwargs(box)
        with torch.inference_mode(),raw._inference_context(),module._sam2_cpu_default_tensor_context():
            pred=raw.predictor;frame_idx=pred.frame_idx
            prepared,_,_=pred.perpare_data(rgb,image_size=pred.image_size)
            image=prepared.cuda().float().unsqueeze(0);backbone=pred.forward_image(image)
            pred.condition_state['cached_features']={frame_idx:(image,backbone)}
            current,_=pred._run_single_frame_inference(output_dict=pred.condition_state['output_dict'],frame_idx=frame_idx,batch_size=pred._get_obj_num(),is_init_cond_frame=False,point_inputs=None,mask_inputs=None,reverse=False,run_mem_encoder=True)
            pred._add_output_per_object(frame_idx,current,'non_cond_frame_outputs')
            for index,outputs in pred.condition_state['output_dict_per_obj'].items():
                outputs['non_cond_frame_outputs'][frame_idx]['object_score_logits']=current['object_score_logits'][index:index+1]
            frame_idx,ids,logits=pred.add_new_prompt(frame_idx=frame_idx,obj_id=raw.placeholder_to_obj_id[role],**kwargs)
            pred.propagate_in_video_preflight()
        masks=module.masks_from_sam2_logits(out_obj_ids=ids,out_mask_logits=logits,obj_id_to_placeholder=raw.obj_id_to_placeholder,image_shape_hw=rgb.shape[:2]);np.savez_compressed(folder/'REPROMPT_MASKS.npz',**masks);points=role_geometry(depth,masks,K,self.T)
        if points.get(role) is None:raise SafetyError('actual SAM re-prompt body has insufficient metric depth')
        write(folder/'RECEIPT.json',{'frame_idx':int(frame_idx),'role':role,'existing_obj_id':raw.placeholder_to_obj_id[role],'all_other_role_ids_preserved':True,'old_frame_stamp_preserved':stamp,'prompted_point_m':points[role].tolist(),'point_is_visible_metric_estimate_not_object_truth':True,'mask_source':'actual_SAM2','frame_sha256':sha})
        # New RGBD is required after actual wait/prompt latency. Never relabel the old frame as fresh.
        return self.capture()
class RelocalizingExecutor(StationaryJudgeExecutor):
    def refresh_lift_attachment(self,stage,samples,obs):
        baseline=[s.q.copy() for s in samples]
        def heartbeat():
            self.guard_limits();current=self.robot.samples();self.safety.samples(current)
            if max(np.max(abs(s.q-q)) for s,q in zip(current,baseline))>.015:raise SafetyError('own robot drifted during body re-grounding')
        obs=self.vision.relocalize(stage['object'],self.output/'CURRENT_BODY_RELOCALIZATION',heartbeat);samples=self.robot.samples();self.safety.samples(samples)
        if obs.roles.get(stage['object']) is None:raise SafetyError('fresh SAM body role missing after actual re-grounding')
        return super().refresh_lift_attachment(stage,samples,obs)
SharedCupExecutor=RelocalizingExecutor
SharedSAMVision=RelocalizingVision
