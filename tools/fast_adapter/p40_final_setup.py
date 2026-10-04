"""Create the independent evaluation freeze, only after all development closes."""
import hashlib,json,time
from pathlib import Path
from tools.fast_adapter.p40_evaluate_v02 import ROOT,STRATA,configurations
TASKS=('cup','hammer','insert')

def binding(p):return dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest())

def main():
    for task in TASKS:
        assert (ROOT/task/'DEV_MODELS_DONE.json').exists(),task
        assert (ROOT/task/'MOTION_DEV_DONE.json').exists(),task
        assert (ROOT/task/'dev_AUDIT.json').exists(),task
        assert (ROOT/task/'MEANS_v03.pt').exists(),task
        assert (ROOT/task/'MEANS_motion.pt').exists(),task
    report=json.loads((ROOT/'summary/dev/RESULTS.json').read_text());strong={t:report['tasks'][t]['strong_fixed'] for t in TASKS};extras={};cells=[]
    for task in TASKS:
        stats=report['tasks'][task]['stats'];extras[task]=[]
        ref=sum(stats[s]['relation/continuous']['success_rate']*8 for s in ('physical','joint'))
        for kind in ('local','fused'):
            value=sum(stats[s][kind+'/continuous']['success_rate']*8 for s in ('physical','joint'))
            if value-ref>=2:extras[task].append(kind)
        for split in STRATA:
            for arm in ('base','fixed0','fixed1','fixed2','fixed3'):cells.append(dict(task=task,split=split,kind='references',seed=0,arm=arm))
            for kind in ('relation','local','fused','motion','history'):
                arms=['continuous']
                if split=='paired':
                    if kind=='relation':arms+=['mean','first12','donor']
                    elif kind=='motion' or kind in extras[task]:arms+=['mean','donor']
                for seed in (11,22):
                    for arm in arms:cells.append(dict(task=task,split=split,kind=kind,seed=seed,arm=arm))
    configs={t:{s:configurations(t,'final',s) for s in STRATA} for t in TASKS}
    source_paths=[Path(p) for p in ('prove/fast_adapter/contact_skills.py','prove/fast_adapter/skill_tokens_v02.py','prove/fast_adapter/skill_visual_reference.py','prove/fast_adapter/skill_observation.py','prove/fast_adapter/planar_tokens.py','prove/fast_adapter/lila_frontend.py','prove/fast_adapter/visual_alignment.py','tools/fast_adapter/p40_evaluate_v02.py','tools/fast_adapter/p40_final_setup.py','tools/fast_adapter/p40_final_queue.py','tools/fast_adapter/p40_audit_v02.py','tools/fast_adapter/p40_report_v02.py')]
    source_paths += [Path(p) for p in ('tools/fast_adapter/p40_train_v03.py','tools/fast_adapter/p40_evaluate.py','tools/fast_adapter/p40_launch_v03.py','data/sources/lila-wam-reference-20260923/models/vla_model_fm.py','assets/dinov3/config.json','assets/dinov3/preprocessor_config.json')]
    source_paths += [ROOT/p for p in ('PROTOCOL.json','TRAIN_FREEZE.json','TRAIN_v02_AMENDMENT.json','TRAIN_v03_AMENDMENT.json','MOTION_TOKEN_PLAN.json','MOTION_FREEZE.json','ANALYSIS_PLAN_v02.json','summary/dev/RESULTS.json')]
    models=[]
    for task in TASKS:
        for kind in ('relation','local','fused','history'):
            for seed in (11,22):models.append(ROOT/task/'models_v03'/kind/f'seed{seed}'/'model.pt')
        for seed in (11,22):models.append(ROOT/task/'motion'/f'seed{seed}'/'model.pt')
        models.extend([ROOT/task/'motion/local.pt',ROOT/task/'motion/PCA.pt',ROOT/task/'effects_v03/local.pt',ROOT/task/('data_v02' if task=='insert' else 'data')/'PCA.pt',ROOT/task/'MEANS_v03.pt',ROOT/task/'MEANS_motion.pt'])
    weights=[Path('assets/dinov3/model.safetensors'),Path('data/sources/lila-wam-reference-20260923/VISUAL_WEIGHTS.pt')]
    for p in weights:assert p.exists(),p
    freeze=dict(time=time.time(),phase='final',status='result-free independent freeze',configs=configs,cells=cells,strong_fixed=strong,optional_visual_controls=extras,
        sources=[binding(p) for p in source_paths],models=[binding(p) for p in models],visual_weights=[binding(p) for p in weights],
        executions=sum(len(configs[c['task']][c['split']]) for c in cells),unique_conditions=288,training_models=30,
        declaration='All five representation/History candidates and both model seeds retained. No best-seed selection. No final results may tune components, exclusions, thresholds or budget.')
    with (ROOT/'FINAL_FREEZE.json').open('x') as f:json.dump(freeze,f,ensure_ascii=False,indent=2)
    print(dict(cells=len(cells),executions=freeze['executions'],strong_fixed=strong,optional=extras))

if __name__=='__main__':main()
