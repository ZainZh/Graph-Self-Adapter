#!/usr/bin/env python3
"""Offline source-contract audit. Does not instantiate an environment or policy."""
from __future__ import annotations
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def audit():
    return {
        'schema': 'prove.fast-adapter-libero-compatibility.v1',
        'scope': 'static source-contract audit; no simulator or pixel qualification',
        'direct_frozen_insert_reuse': False,
        'libero_placements': {name: 'unsupported frozen Insert semantics'
                              for name in ('book_to_caddy', 'bowl_to_drawer', 'mug_to_microwave')},
        'checklist': [
            {'requirement': 'four visible coplanar goal corners', 'existing': 'required for PnP/ray objective',
             'libero': 'no general public corner producer qualified for these placement targets'},
            {'requirement': 'calibrated wrist RGB at 448x448', 'existing': '448 coordinates hardcoded in features',
             'libero': 'optional public RGB-D calibration exists; exact 448 alignment not qualified'},
            {'requirement': 'actor/object/goal visible masks', 'existing': 'rendered segmentation masks and role boxes',
             'libero': 'basic public RGB/state and optional RGB-D do not supply these role masks'},
            {'requirement': 'own EEF pose', 'existing': 'world position plus 3x3 rotation',
             'libero': 'public proprioception exists; coordinate and rotation conversion needs qualification'},
            {'requirement': 'task domain', 'existing': 'cup handle, rigid preheld peg insertion, sliding handle',
             'libero': 'placement in caddy/drawer/microwave is outside frozen task semantics'},
            {'requirement': 'controller format', 'existing': 'absolute world position, rotation matrix, closed flag',
             'libero': 'existing runner executes clipped 7D environment actions; no legacy controller bridge qualified'},
            {'requirement': 'latent lifetime', 'existing': 'z persists within learner; support cap 32; new learner resets',
             'libero': 'new stage wrapper resets z/reference/support and preserves controller lock semantics'},
        ],
        'reusable_now': 'frozen estimator lifecycle on compatible public observations, not arbitrary LIBERO execution',
        'sources': ['prove/fast_adapter/p45_neural.py', 'prove/fast_adapter/p47_neural.py',
                    'prove/fast_adapter/neural_geometry_adapter.py',
                    'prove/fast_adapter/robot_geometry_env_p45.py',
                    'tools/fast_adapter/p45_common.py', 'prove/rollout/libero_observation.py',
                    'tools/rollout/audit_pi05_libero_action_semantics.py'],
    }


if __name__ == '__main__':
    print(json.dumps(audit(), indent=2))
