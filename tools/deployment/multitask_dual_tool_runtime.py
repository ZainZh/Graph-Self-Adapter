"""Runtime leaf for V22 routes; behavior stays in the reviewed specialist executor."""
import copy

from prove.xtrainer.io import SafetyError
from multitask_special_runtime import SpecialExecutor
from shared_cup_vision import SharedSAMVision


class DualToolExecutor(SpecialExecutor):
    def __init__(self, bundle, *args, **kwargs):
        if bundle["graph"].get("schema") != "graph-self-adapter.multitask-dual-tool-route.v22":
            raise SafetyError("explicit V22 dual-tool route required")
        task = bundle.get("task")
        if task not in ("sorting_cube_bimanual", "sweep_objects"):
            raise SafetyError("unqualified dual-tool task cannot enter runtime")
        local = copy.copy(bundle)
        local["graph"] = copy.deepcopy(bundle["graph"])
        local["graph"]["schema"] = "graph-self-adapter.multitask-special-route.v1"
        super().__init__(local, *args, **kwargs)
        self.b["graph"]["schema"] = "graph-self-adapter.multitask-dual-tool-route.v22"
