import api_v28_cup_entry as parent
from pathlib import Path
HERE=Path(__file__).resolve().parent
parent.parent.entry.SOURCES += (Path(__file__),HERE/'sim_preview_v33.py',HERE/'task_stream_view_v33.py',HERE/'three_task_v33.py')
def main(argv=None):return parent.main(argv)
