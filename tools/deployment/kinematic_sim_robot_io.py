"""Candidate simulator I/O with FK limited to MuJoCo kinematics."""
from __future__ import annotations

import numpy as np

from fair_sim_robot_io import FairSimRobot, SimRGBDCamera, SimStop
from prove.xtrainer.io import array


class KinematicFairSimRobot(FairSimRobot):
    def fk(self, q, g, arm):
        q = array(q, (2, 6)); g = array(g, (2,)); index = 0 if arm == "left" else 1
        with self.fk_lock:
            self.scratch.qpos[:] = self.world.model.qpos0
            self.scratch.qpos[self.world.arm_qpos_indices] = q.reshape(-1)
            travel = float(self.world.model.numeric("gripper_closing_travel").data[0])
            for joint, _, finger_index, sign in self.world.fingers:
                self.scratch.qpos[joint.qposadr[0]] = sign * travel * (1 - g[finger_index])
            self.mj.mj_kinematics(self.world.model, self.scratch)
            site = self.scratch.site(arm + "_tcp")
            xyz = site.xpos.copy(); rotation = site.xmat.reshape(3, 3).copy()
        if index:
            return (self.inverse @ np.r_[xyz, 1])[:3], self.inverse[:3, :3] @ rotation
        return xyz, rotation


SimRobot = KinematicFairSimRobot
