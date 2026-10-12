"""Show public sealed execution outcomes explicitly, without reading private evaluation."""
import json
from task_stream_view_v33 import StreamingTaskView

class ClearStateView(StreamingTaskView):
    def status(self):
        status=super().status()
        if self.run:
            seal=self.run/'PREDICTIONS_SEALED.json'
            if seal.exists():
                try:
                    public=json.loads(seal.read_text()).get('predictions',{})
                    if public:
                        completed=sum(bool(x.get('completed')) for x in public.get('stage_trace',[]))
                        status.update(completed_nodes=completed,total_nodes=public.get('graph_stages'),stop_reason=public.get('failure'))
                        if public.get('failure'):
                            status.update(execution_state='failed',state_label='执行失败 · 未完成整张图',message=public['failure'])
                        elif public.get('graph_completed'):
                            status.update(execution_state='finished',state_label='任务图执行完成',message='所有图节点执行完成；完整验收以封存报告为准。')
                        else:status.update(state_label='已停止 · 未确认整图完成')
                except (OSError,ValueError):pass
        return status

def install():
    import task_live_view_v27
    task_live_view_v27.TaskLiveView=ClearStateView
