"""Feature-only compatibility inspection of public LiLa-WAM, no training/control."""
from pathlib import Path
import json,importlib.util,urllib.request
import torch
import torch.nn.functional as F
from prove.fast_adapter.visual_alignment import FrozenDinoROI,DEFAULT_MODEL
from tools.fast_adapter.p18_run import save_json,digest
R=Path('evaluations/fast_adapter_p20_insert3d_v01')
O=R/'diagnostics/lila_wam_compatibility_v01'
S=Path('data/sources/lila-wam-reference-20260923')

def main():
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;O.mkdir(parents=True,exist_ok=False)
    metadata=json.loads(Path('/tmp/p20_lila_modelscope_0.json').read_text())
    remote=next(f for f in metadata['Data']['Files'] if f['Path']=='checkpoint.pt')
    checkpoint=Path('/home/zheng/github/LiLa-WAM/checkpoints_vla/public/checkpoint.pt')
    checkpoint_sha=digest(checkpoint);assert checkpoint_sha==remote['Sha256']
    save_json(O/'OFFICIAL_MODEL_FILES.json',metadata)
    config_url='https://modelscope.cn/api/v1/models/yangfan97/LiLa-WAM_RoboTwin2_0/repo?Revision=master&FilePath=config.yaml'
    config=urllib.request.urlopen(config_url,timeout=30).read();(O/'OFFICIAL_CONFIG.yaml').write_bytes(config)
    config_meta=next(f for f in metadata['Data']['Files'] if f['Path']=='config.yaml');assert digest(O/'OFFICIAL_CONFIG.yaml')==config_meta['Sha256']
    spec=importlib.util.spec_from_file_location('official_lila',S/'models/vla_model_fm.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    state=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)['model_state_dict']
    fusion=m.MultiLayerConcatFusion(1024,3,1024,proj_type='linear',pre_norm=True).cuda().eval()
    adapter=m.VisualFeatureAdapter(1024,768,num_queries=64,num_heads=8,num_layers=4,dropout=0.).cuda().eval()
    def subset(prefix):return {k[len(prefix):]:v for k,v in state.items() if k.startswith(prefix)}
    fload=fusion.load_state_dict(subset('concat_fusion.'),strict=True);aload=adapter.load_state_dict(subset('dino_adapters.0.'),strict=True)
    fusion.requires_grad_(False);adapter.requires_grad_(False)
    rgb_source=R/'diagnostics/dinov3_features_v01/FEATURES.pt';rgb=torch.load(rgb_source,map_location='cpu',weights_only=False)['rgb']
    encoder=FrozenDinoROI(DEFAULT_MODEL,'cuda');records=[];tokens={}
    with torch.inference_mode():
        for h,w in [(224,224),(240,320)]:
            pixels=rgb.cuda().permute(0,3,1,2).float()/255
            if (h,w)!=(224,224):pixels=F.interpolate(pixels,size=(h,w),mode='bilinear',align_corners=False)
            inputs=((pixels-encoder.mean)/encoder.std).to(torch.bfloat16)
            outputs=encoder.model(pixel_values=inputs,output_hidden_states=True,return_dict=True)
            assert len(outputs.hidden_states)==25
            selected=[outputs.hidden_states[i].float() for i in [-12,-8,-4]]
            fused=fusion(selected);out=adapter(fused)
            assert out.shape==(3,64,768) and torch.isfinite(out).all()
            assert not torch.equal(out[0],out[1]) and not torch.equal(out[1],out[2])
            name=f'{w}x{h}';tokens[name]=out.cpu()
            records.append({'input_size_wh':[w,h],'input_tokens':selected[0].shape[1],'selected_hidden_state_indices':[-12,-8,-4],'equivalent_blocks':[13,17,21],'keep_cls_and_register':True,'fused_shape':list(fused.shape),'adapter_shape':list(out.shape),'finite':True,'different_images_produce_different_outputs':True,'resized_from_same224_RGB':(h,w)!=(224,224)})
    torch.save(tokens,O/'EXTRACTED_VISUAL_TOKENS.pt')
    receipt={'status':'feature_only_compatibility_passed_not_efficacy_test','source_commit':json.loads((S/'SOURCE.json').read_text())['commit'],'modelscope_repository':'https://www.modelscope.cn/models/yangfan97/LiLa-WAM_RoboTwin2_0','checkpoint_path':str(checkpoint),'checkpoint_sha256':checkpoint_sha,'official_checkpoint_sha256_match':True,'config_sha256':digest(O/'OFFICIAL_CONFIG.yaml'),'strict_fusion_load':str(fload),'strict_adapter_load':str(aload),'module_parameter_count':sum(p.numel() for p in fusion.parameters())+sum(p.numel() for p in adapter.parameters()),'source_rgb':str(rgb_source),'source_rgb_sha256':digest(rgb_source),'checks':records,'inputs_used':['RGB only'],'no_action_proprio_task_or_future_input_to_this_frontend':True,'frozen_weights':True,'no_training_or_rollout':True,'not_established':['better geometric encoding','object-separated slots','better insertion success','test-time gradient advantage'],'important_difference':'Official current checkpoint/config fuse hidden_states[-12,-8,-4], i.e. blocks13/17/21. Paper describes17/21 only. Weight shape3072 confirms3 layers; exact identities require config, not tensor shape alone.','output_tokens_sha256':digest(O/'EXTRACTED_VISUAL_TOKENS.pt')}
    save_json(O/'COMPATIBILITY.json',receipt);print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
