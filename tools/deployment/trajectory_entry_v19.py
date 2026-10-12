#!/usr/bin/env python3
"""Same V18 motion; V19 frozen compact OpenRouter supported-boundary judge."""
import hashlib
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
for path in (HERE,REPO):
    if str(path) not in sys.path:sys.path.insert(0,str(path))
import trajectory_entry_v18 as motion
from gemini_stage_judge import GeminiStageJudge,validate_config,MODEL_IDENTITY
from prove.xtrainer.agent import digest,write

base=motion.base;TASK=motion.TASK;VARIANT='white_stage_trajectory_v19_gemini_judge'
SOURCES=(HERE/'gemini_stage_judge.py',HERE/'trajectory_entry_v19.py',HERE/'compact_stage_judge.py',REPO/'tests/test_compact_stage_judge.py',REPO/'tests/test_gemini_stage_judge.py')

def runtime_hash():return hashlib.sha256((motion.runtime_hash()+''.join(digest(path) for path in SOURCES)).encode()).hexdigest()

def bindings(config_path,cfg,mode):
    validate_config(cfg['runtime']['stage_judge']);report,effective,bundle,bound=motion.bindings(config_path,cfg,mode)
    bound.update({str(path.resolve()):digest(path) for path in SOURCES})
    protocol=cfg['runtime']['stage_judge'].get('protocol_file')
    if protocol:bound[str(Path(protocol).resolve())]=digest(protocol)
    return report,effective,bundle,bound

def guard(*args,**kwargs):return base.guard(*args,**kwargs)

def install(config):
    from types import SimpleNamespace
    judge_config=config['runtime']['stage_judge'];validate_config(judge_config)
    def factory(*args,**kwargs):return GeminiStageJudge(*args,**kwargs,key_file=judge_config['key_file'],maximum_calls=judge_config['maximum_calls'])
    def load(path,name):
        module=motion.load(path,name)
        if Path(path).name!='direct_white_runtime.py':return module
        class Executor(module.SharedCupExecutor):
            def run(self):
                result=super().run();result.update(execution_variant=VARIANT,stage_judge_backend='OpenRouter',stage_judge_identity=MODEL_IDENTITY,stage_judge_API_records=self.judge.api_records,initial_Grounding_source='current Root manual RGB role boxes',motion_runtime_same_as_v18=True)
                write(self.output/'RESULT.json',result);return result
        return SimpleNamespace(SharedCupExecutor=Executor,SharedSAMVision=module.SharedSAMVision)
    base.bindings=bindings;base.runtime_hash=runtime_hash;base.load=load;base.VARIANT=VARIANT;base.RealRootJudge=factory

def main(argv=None):
    import argparse,json
    p=argparse.ArgumentParser(add_help=False);p.add_argument('--config',type=Path,required=True);a,_=p.parse_known_args(argv)
    install(json.loads(a.config.read_text()));return base.main(argv)

if __name__=='__main__':main()
