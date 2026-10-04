"""P30 held-tool reset for isolating subsequent tool/target interaction.

Privileged preparation is environment initialization, never learner feedback.
The free hammer remains physically gripped, without a weld or external force.
"""
import numpy as np
from prove.fast_adapter.object_interaction_env import ObjectEnv


class PreparationFailed(RuntimeError):
    pass


class HeldHammerEnv(ObjectEnv):
    def __init__(self, cfg, render=True, *, preparation_budget=120,
                 required_streak=5, required_lift=.055):
        if cfg['task'] != 'hammer-block':
            raise ValueError('Held-tool initialization is only for the hammer task')
        super().__init__(cfg, render=render)
        self.preparation_budget = preparation_budget
        self.required_streak = required_streak
        self.required_lift = required_lift
        self.preparation = None

    def reset(self, seed):
        # Base reset has no self.step dispatch, so preparation actions remain
        # separate from the fixed-closed-gripper task execution below.
        super().reset(seed)
        grip = self.env.data.site('grip_site').xpos.copy()
        lift_goal = grip + np.array([0., 0., .095])
        actions, metrics, states = [], [], [self.state()]
        streak = 0
        for _ in range(self.preparation_budget):
            if self.phase < 3:
                action = super().teacher()
            else:
                delta = lift_goal - self.env.tcp_center
                action = np.r_[np.clip(delta * 15, -1, 1), .9].astype(np.float32)
            super().step(action)
            m = dict(self.metrics[-1])
            actions.append(action.copy()); metrics.append(m); states.append(self.state())
            valid = bool(m['bilateral_handle'] and m['lift'] >= self.required_lift)
            streak = streak + 1 if valid else 0
            contaminated = any(x['head_target_contact'] or x['target_direct_contact'] or
                               abs(x['target_displacement']) > .001 for x in metrics)
            if contaminated or streak >= self.required_streak:
                break
        self.preparation = dict(seed=int(seed), actions=actions, metrics=metrics,
                                states=states, steps=len(actions),
                                passed=streak >= self.required_streak and not contaminated)
        if not self.preparation['passed']:
            raise PreparationFailed('Stable held-tool initialization failed; retain this condition')
        self.metrics = []
        self.phase, self.wait = 3, 0
        e = self.env
        e.hold_streak = 0
        e.head_contact_seen = e.direct_contact_seen = False
        e.handle_contact_seen = True
        self.last_executed_action = np.array([0., 0., 0., .9], dtype=np.float32)
        return self.public()

    def step(self, action):
        actual = np.clip(np.asarray(action, dtype=np.float32), -1, 1).copy()
        actual[3] = .9
        self.last_executed_action = actual
        return super().step(actual)

    def teacher(self):
        """Geometry teacher for offline training/qualification ONLY.

        Reach the target's near face with the tool head; ground-truth geometry
        must not be used by an online adapter or learned action policy.
        """
        e = self.env
        tip = e.data.site('tool_site').xpos.copy()
        target = e.data.site('target_site').xpos.copy()
        desired = target.copy()
        desired[1] += .055 - float(e.data.joint('target_slide').qpos[0])
        if tip[2] < target[2] - .008:
            delta = np.array([0., 0., target[2] + .006 - tip[2]])
        else:
            delta = desired - tip
        return np.r_[np.clip(delta * 15, -1, 1), .9].astype(np.float32)
