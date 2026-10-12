import cup_v25_current_goalset_one_command as parent
from api_grounding_v32 import install
from pathlib import Path
install('put_mug_on_box',parent)
old_load=parent.entry.load

def load(path,name):
    if Path(path).name=='direct_white_runtime.py':
        import cup_visible_plane_v36 as runtime
        return runtime
    return old_load(path,name)
parent.entry.load=load
HERE=Path(__file__).resolve().parent
parent.entry.SOURCES+=(Path(__file__),HERE/'cup_visible_plane_v36.py',HERE/'sim_preview_v33.py',HERE/'task_stream_view_v33.py',HERE/'task_stream_view_v35.py',HERE/'three_task_v36.py')
def main(argv=None):return parent.main(argv)
