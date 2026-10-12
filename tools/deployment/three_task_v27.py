#!/usr/bin/env python3
"""Common task launcher with a separate read-only public-camera/stage viewer."""
import argparse,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[1]
for p in (HERE,REPO):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
PORTS={'put_mug_on_box':18734,'collect_apple':18735,'collect_cubes':18736}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--task',choices=tuple(PORTS),required=True);p.add_argument('--backend',choices=('sim','real'),default='sim');p.add_argument('--live-port',type=int);p.add_argument('--view-only',action='store_true');p.add_argument('--run-output',type=Path)
    args,rest=p.parse_known_args(argv)
    import three_task_v26 as parent
    invocation=['--task',args.task,'--backend',args.backend]+rest
    if not args.view_only and (args.backend=='real' or '--check' in rest or '--prepare-only' in rest):return parent.main(invocation)
    if args.view_only and (args.backend!='sim' or args.run_output is None or not args.run_output.is_dir()):p.error('--view-only requires an existing --run-output and sim backend')
    if not args.view_only:
        # Reject disabled candidates before opening a viewer or reserving an episode.
        reg=parent.read_registry(parent.DEFAULT_REGISTRY)
        if reg['tasks'][args.task].get('default_execution_enabled') is False and '--experimental' not in rest:p.error('this task is not qualified; use --check or explicit --experimental')
    from task_live_view_v27 import TaskLiveView
    view=TaskLiveView(args.task,args.live_port or PORTS[args.task])
    if args.run_output:view.attach(args.run_output)
    try:url=view.start()
    except OSError as e:p.error('实时查看端口不可用，请用 --live-port 指定空闲端口：'+str(e))
    print(json.dumps({'task':args.task,'live_view_url':url,'read_only':True,'extra_camera_captures':0},ensure_ascii=False),flush=True)
    if args.view_only:
        try:
            while True:time.sleep(1)
        except KeyboardInterrupt:pass
        finally:view.close()
        return
    import three_task_v26_operator_scope as scopes
    original=scopes.new_operator_scope
    def scope(*a,**kw):
        result=original(*a,**kw);view.set_owner(result['fresh_owner']);return result
    scopes.new_operator_scope=scope
    # Explicit scopes also carry an exact task owner; no guessing historical runs.
    for i,arg in enumerate(rest):
        if arg=='--scope' and i+1<len(rest):view.set_owner(Path(rest[i+1]).resolve().parent)
        elif arg.startswith('--scope='):view.set_owner(Path(arg.split('=',1)[1]).resolve().parent)
    try:return parent.main(invocation)
    finally:
        scopes.new_operator_scope=original;view.finished=True;view.refresh();view.close()

if __name__=='__main__':main()
