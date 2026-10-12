"""Hanging-mug V22.2: table patch is optional; mug and rack stay mandatory."""
from __future__ import annotations

from prove.xtrainer.io import SafetyError
from multitask_articulated_v22 import ArticulatedExecutorV22


def optional_table_role_contract(contract: dict) -> dict:
    result = {key: list(value) if isinstance(value, list) else value for key, value in contract.items()}
    required = result.get("SAM_roles", [])
    if "mug" not in required or "rack" not in required:
        raise SafetyError("hanging actor/target role contract missing")
    result["SAM_roles"] = [role for role in required if role != "table_patch"]
    result["optional_SAM_roles"] = ["table_patch"]
    result["optional_role_reason"] = "auxiliary observed patch is not read by hanging policy, target grounding, holding, or exact goal judge"
    return result


class HangingExecutorV22_2(ArticulatedExecutorV22):
    def __init__(self, bundle, *args, **kwargs):
        if bundle.get("task") != "hanging_mug":
            raise SafetyError("V22.2 repair is hanging_mug only")
        super().__init__(bundle, *args, **kwargs)
        self.roles = optional_table_role_contract(self.roles)

    def run(self):
        result = super().run()
        result.update(
            execution_variant="hanging_mug_optional_aux_table_v22_2",
            required_start_roles=self.roles["SAM_roles"],
            optional_start_roles=self.roles["optional_SAM_roles"],
            actor_target_freshness_unchanged=True,
        )
        from prove.xtrainer.agent import write
        write(self.output / "RESULT.json", result)
        return result


SharedCupExecutor = HangingExecutorV22_2
