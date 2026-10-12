"""Apply the verified empty-gripper clearance relation to both release boundaries."""
import json,copy
from cup_clear_goalset_v41 import SharedCupExecutor as BaseExecutor,SharedSAMVision
class BothClearExecutor(BaseExecutor):
    def _code_goals_and_gripper(self,stage,pose,obs):
        if stage['stage_id']=='white_direct.08b.clear_retreat':
            delegated=copy.deepcopy(stage);delegated['stage_id']='c02.07b.clear_retreat'
            result=super()._code_goals_and_gripper(delegated,pose,obs)
            record=self.output/'EMPTY_GRIPPER_CLEARANCE_GOALSET.json'
            if record.exists():(self.output/'WHITE_EMPTY_GRIPPER_CLEARANCE_GOALSET.json').write_bytes(record.read_bytes())
            return result
        return super()._code_goals_and_gripper(stage,pose,obs)
SharedCupExecutor=BothClearExecutor
