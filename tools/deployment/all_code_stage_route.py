"""Exact controller ownership for the 25-node cup ALL_CODE_STAGE candidate."""
from prove.xtrainer.io import SafetyError


CODE_STAGE_IDS = frozenset({
    "c01.01.approach", "c01.03.lift", "c01.04.move", "c01.06.release", "c01.07.wait",
    "c02.01.approach", "c02.03.lift", "c02.04.move", "c02.06.release", "c02.07.wait",
    "white_direct.01.approach", "white_direct.03.lift", "white_direct.04.raise",
    "white_direct.05.move", "white_direct.06.lower", "white_direct.07.release",
    "white_direct.08.retreat",
})
PRECISION_STAGE_IDS = frozenset({
    "c01.02.grasp", "c01.05.align", "c02.02.grasp", "c02.05.align",
    "white_direct.02.grasp",
})
SEMANTIC_STAGE_IDS = frozenset({
    "c01.08.complete", "c02.08.complete", "white_direct.09.complete",
})


def validate_route(graph):
    stages = {stage["stage_id"]: stage for stage in graph["stages"]}
    expected = CODE_STAGE_IDS | PRECISION_STAGE_IDS | SEMANTIC_STAGE_IDS
    if set(stages) != expected or len(stages) != 25:
        raise SafetyError("ALL_CODE_STAGE requires the exact frozen 25-node V19 graph")
    for stage_id in CODE_STAGE_IDS:
        stage = stages[stage_id]
        if stage["cycle_id"] in ("c01", "c02") and stage["controller"] != "code_reference_retarget":
            raise SafetyError("red Code controller ownership changed: " + stage_id)
        if stage["cycle_id"] == "white_direct" and stage["controller"] != "observed_white_direct_Code":
            raise SafetyError("white Code controller ownership changed: " + stage_id)
    for stage_id in PRECISION_STAGE_IDS:
        stage = stages[stage_id]
        if stage["cycle_id"] in ("c01", "c02") and stage["controller"] != "neural_short":
            raise SafetyError("red precision controller ownership changed: " + stage_id)
        if stage["cycle_id"] == "white_direct" and stage["phase"] != "Grasp":
            raise SafetyError("white precision ownership changed: " + stage_id)
    if any(stages[stage_id]["controller"] != "supported_Root_completion" for stage_id in SEMANTIC_STAGE_IDS):
        raise SafetyError("supported semantic boundary ownership changed")
    return {
        "schema": "graph-self-adapter.all-code-stage-route.v1",
        "graph_nodes": 25,
        "code_trajectory_stage_ids": sorted(CODE_STAGE_IDS),
        "precision_closed_loop_stage_ids": sorted(PRECISION_STAGE_IDS),
        "semantic_boundary_stage_ids": sorted(SEMANTIC_STAGE_IDS),
        "fixed_z": True,
        "graph_or_weight_bytes_modified": False,
    }

