#!/usr/bin/env python3
"""Reusable task launch. Each invocation owns a fresh scope; no consumed reservation is reused."""
import argparse,importlib.util,json,sys,uuid
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
for path in (HERE,REPO):
 if str(path) not in sys.path:sys.path.insert(0,str(path))
REGISTRY={
 'collect_apple':('three_task_v35.py','collect_apple',None,'原苹果入碗'),
 'put_mug_on_box':('three_task_v44.py','put_red_mug_on_box',None,'简化为单红杯放箱'),
 'put_red_mug_on_box':('three_task_v44.py','put_red_mug_on_box',None,'单红杯放箱'),
 'sort_two_spoon':('task_auto_v72.py','put_black_spoon_on_plate',None,'简化为单黑勺放盘'),
 'put_black_spoon_on_plate':('task_auto_v72.py','put_black_spoon_on_plate',None,'单黑勺放盘'),
 'stack_cups':('night_tasks_v51.py','nest_two_cups','night','简化为黄杯套蓝杯'),
 'nest_two_cups':('night_tasks_v51.py','nest_two_cups','night','黄杯套蓝杯'),
 'open_drawer_and_place_object':('night_tasks_v56.py','open_drawer_only','night','简化为只开抽屉'),
 'open_drawer_only':('night_tasks_v56.py','open_drawer_only','night','只开抽屉'),
 'collect_cubes':('night_tasks_v59.py','collect_single_cube','night','简化为单黄方块入碗'),
 'collect_single_cube':('night_tasks_v59.py','collect_single_cube','night','单黄方块入碗'),
 'sorting_cube_bimanual':('night_tasks_v67.py','sort_one_yellow_cube','night','单黄方块双臂交接并放入托盘'),
 'sort_one_yellow_cube':('night_tasks_v67.py','sort_one_yellow_cube','night','单黄方块双臂交接并放入托盘'),
 'sweep_objects':('night_tasks_v68.py','sweep_with_stationary_pan','night','简化为单扫刷清扫；簸箕在桌面自由放置'),
 'sweep_with_stationary_pan':('night_tasks_v68.py','sweep_with_stationary_pan','night','单扫刷清扫到桌面簸箕')}
def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=tuple(REGISTRY));p.add_argument('--backend',choices=('sim','real'),default='sim');p.add_argument('--live-port',type=int);p.add_argument('--list',action='store_true');p.add_argument('--check',action='store_true');a=p.parse_args(argv)
 if a.list:print(json.dumps({k:r[3] for k,r in REGISTRY.items()},ensure_ascii=False,indent=2));return
 if not a.task:p.error('--task is required')
 script,logical,kind,label=REGISTRY[a.task]
 if a.backend=='real' and not a.check:p.error('当前只交付模拟启动和真实接口离线检查，未验收真实硬件执行')
 if a.check:
  print(json.dumps({'task':a.task,'actual_goal':label,'entry':str(HERE/script),'fresh_invocation_scope':True,'hardware_qualified':False,'episode_started':False},ensure_ascii=False,indent=2));return
 args=['--task',logical,'--backend','sim']
 if kind=='night':args+=['--candidate','operator_'+uuid.uuid4().hex[:12]]
 import socket
 ports={'collect_apple':18735,'put_red_mug_on_box':18734,'put_black_spoon_on_plate':18743,'nest_two_cups':18802,'open_drawer_only':18800,'collect_single_cube':18806,'sweep_with_stationary_pan':18807,'sort_one_yellow_cube':18803};port=a.live_port or ports[logical]
 probe=socket.socket()
 try:probe.bind(('127.0.0.1',port))
 except OSError:
  probe.close();probe=socket.socket();probe.bind(('127.0.0.1',0));port=probe.getsockname()[1]
 finally:probe.close()
 args+=['--live-port',str(port)]
 print('执行目标：'+label,flush=True);spec=importlib.util.spec_from_file_location('_selected_task_entry',HERE/script);module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 if script=='three_task_v35.py':return module.base.base.base.main(args)
 if script=='three_task_v44.py':
  args=['put_mug_on_box' if x=='put_red_mug_on_box' else x for x in args];return module.base.base.base.base.main(args)
 return module.main(args)
if __name__=='__main__':main()
