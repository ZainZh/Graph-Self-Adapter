#!/usr/bin/env python3
"""Build remote-only dual-tool manifest/config derivatives from shared bindings."""
import argparse,hashlib,json
from pathlib import Path


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def rewrite(value,replacements):
    if isinstance(value,dict):return {rewrite(k,replacements):rewrite(v,replacements) for k,v in value.items()}
    if isinstance(value,list):return [rewrite(v,replacements) for v in value]
    if isinstance(value,str):
        for a,b in replacements:value=value.replace(a,b)
    return value


def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--mujoco',type=Path,required=True);p.add_argument('--robort',type=Path,required=True);p.add_argument('--sam',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True);a=p.parse_args(argv)
    repo=a.repo.resolve();owner=repo/'evaluations/multitask_parallel_v22/dual_tool/server_shared';owner.mkdir(parents=True,exist_ok=True)
    repl=(("/home/zheng/github/Graph-Self-Adapter",str(repo)),("/home/zheng/github/Mujoco-RSI",str(a.mujoco.resolve())),("/home/zheng/github/RoboTwin_bin_pick",str(a.robort.resolve())),("/home/zheng/github/SAM2_streaming",str(a.sam.resolve())),("/home/zheng/Datasets/sam2/sam2.1_hiera_small.pt",str(a.checkpoint.resolve())),("/home/zheng/miniforge3/envs/geometry/bin/python","/home/ubuntu/venvs/graph-v22/bin/python"))
    shared=repo/'evaluations/multitask_parallel_v22/server_shared';inventory=json.loads((shared/'DUAL_TOOL_INVENTORY_SERVER.json').read_text());records=[]
    catalogs={'sorting_cube_bimanual':shared/'EXTERNAL_ASSET_BINDINGS_V02_SERVER.json','sweep_objects':shared/'EXTERNAL_ASSET_BINDINGS_V01_SERVER.json'}
    for row in inventory['tasks']:
        task=row['task']
        if task not in catalogs:continue
        task_owner=repo/'evaluations/multitask_parallel_v22/dual_tool'/task/'server_fair_v22_1';solution=task_owner/'solution_derivative';solution.mkdir(parents=True,exist_ok=True)
        manifest_source=Path(row['bindings']['manifest']['path']);manifest=solution/'MANIFEST.json';manifest.write_text(json.dumps(rewrite(json.loads(manifest_source.read_text()),repl),indent=2,sort_keys=True)+'\n')
        records.append({'source':str(manifest_source),'source_sha256':digest(manifest_source),'target':str(manifest),'target_sha256':digest(manifest)})
        row['bindings']['manifest']={'path':str(manifest),'sha256':digest(manifest)};row['solution']=str(solution)
        external_source=Path(row['solution']).parent/'EXTERNAL_BINDINGS.json' if False else None
        original_solution=Path(rewrite(json.loads((shared/'DUAL_TOOL_INVENTORY_SERVER.json').read_text()),repl)['tasks'][[r['task'] for r in inventory['tasks']].index(task)]['solution'])
        candidate=original_solution/'EXTERNAL_BINDINGS.json'
        if candidate.is_file():
            target=solution/'EXTERNAL_BINDINGS.json';target.write_text(json.dumps(rewrite(json.loads(candidate.read_text()),repl),indent=2,sort_keys=True)+'\n')
            records.append({'source':str(candidate),'source_sha256':digest(candidate),'target':str(target),'target_sha256':digest(target)})
        config_source=repo/'evaluations/multitask_parallel_v22/dual_tool'/task/'CONFIG_PROPOSED.json';config=rewrite(json.loads(config_source.read_text()),repl)
        config.update(external_asset_catalog=str(catalogs[task]),server_asset_inventory=str(owner/'DUAL_TOOL_INVENTORY_SERVER.json'),server_shared_manifest=str(shared/'REMOTE_DERIVATIVE_MANIFEST.json'))
        config['runtime']['stage_judge']['key_file']='/home/ubuntu/.config/graph-self-adapter/openrouter.key'
        target_config=task_owner/'CONFIG_SERVER.json';target_config.write_text(json.dumps(config,indent=2,sort_keys=True)+'\n')
        records.append({'source':str(config_source),'source_sha256':digest(config_source),'target':str(target_config),'target_sha256':digest(target_config)})
    target_inventory=owner/'DUAL_TOOL_INVENTORY_SERVER.json';target_inventory.write_text(json.dumps(inventory,indent=2,sort_keys=True)+'\n')
    records.append({'source':str(shared/'DUAL_TOOL_INVENTORY_SERVER.json'),'source_sha256':digest(shared/'DUAL_TOOL_INVENTORY_SERVER.json'),'target':str(target_inventory),'target_sha256':digest(target_inventory)})
    for record in records:
        if '/home/zheng/' in Path(record['target']).read_text():raise RuntimeError('unmapped path in '+record['target'])
    manifest={'schema':'graph-self-adapter.dual-tool-server-derivatives.v1','shared_manifest_sha256':digest(shared/'REMOTE_DERIVATIVE_MANIFEST.json'),'records':records,'historical_sources_modified':False}
    path=owner/'REMOTE_DUAL_TOOL_MANIFEST.json';path.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n');print(json.dumps({'manifest':str(path),'sha256':digest(path),'records':len(records)}))
if __name__=='__main__':main()
