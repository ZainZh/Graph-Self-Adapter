import numpy as np
import pytest
import os

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import importlib.util
from pathlib import Path
import sys

MODULE_PATH = Path(__file__).parents[1] / "prove" / "task_graph" / "twin_physics.py"
SPEC = importlib.util.spec_from_file_location("_twin_physics_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
TWIN = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TWIN
SPEC.loader.exec_module(TWIN)
RoleObservation = TWIN.RoleObservation
TwinWorld = TWIN.TwinWorld
VisibleRGBDFrame = TWIN.VisibleRGBDFrame
_array = TWIN._array
_load_npz = TWIN._load_npz


def _mini_model():
    chain = lambda arm: f"""
      <body name="{arm}_base" pos="{'-.3' if arm == 'left' else '.3'} 0 .5">
        <body name="{arm}_link1"><joint name="{arm}_j1" type="hinge" axis="1 0 0"/><geom type="sphere" size=".005" mass=".01" contype="0" conaffinity="0"/>
        <body name="{arm}_link2" pos="0 0 .03"><joint name="{arm}_j2" type="hinge" axis="0 1 0"/><geom type="sphere" size=".005" mass=".01" contype="0" conaffinity="0"/>
        <body name="{arm}_link3" pos="0 0 .03"><joint name="{arm}_j3" type="hinge" axis="0 0 1"/><geom type="sphere" size=".005" mass=".01" contype="0" conaffinity="0"/>
        <body name="{arm}_link4" pos="0 0 .03"><joint name="{arm}_j4" type="hinge" axis="1 0 0"/><geom type="sphere" size=".005" mass=".01" contype="0" conaffinity="0"/>
        <body name="{arm}_link5" pos="0 0 .03"><joint name="{arm}_j5" type="hinge" axis="0 1 0"/><geom type="sphere" size=".005" mass=".01" contype="0" conaffinity="0"/>
        <body name="{arm}_link6" pos="0 0 .03"><joint name="{arm}_j6" type="hinge" axis="0 0 1"/><geom type="sphere" size=".005" mass=".01" contype="0" conaffinity="0"/>
          <site name="{arm}_tcp" pos="0 0 .03"/>
        </body></body></body></body></body></body>
        <body name="{arm}_finger1"><joint name="{arm}_gripper1" type="slide" axis="1 0 0" limited="true" range="-.04 .04"/><geom type="sphere" size=".01"/></body>
        <body name="{arm}_finger2"><joint name="{arm}_gripper2" type="slide" axis="1 0 0" limited="true" range="-.04 .04"/><geom type="sphere" size=".01"/></body>
      </body>"""
    motors = "\n".join(
        f'<motor name="{arm}_motor{i}" joint="{arm}_j{i}"/>'
        for arm in ("left", "right") for i in range(1, 7)
    ) + "\n" + "\n".join(
        f'<motor name="{arm}_gripper_motor{i}" joint="{arm}_gripper{i}"/>'
        for arm in ("left", "right") for i in (1, 2)
    )
    xml = f"""<mujoco><option timestep=".001" gravity="0 0 -9.81"/>
    <custom><numeric name="gripper_closing_travel" data=".03"/></custom>
    <worldbody>
      <geom name="table" type="plane" size="2 2 .1"/>
      {chain('left')}{chain('right')}
      <body name="object_apple" pos="0 0 .08"><freejoint/><geom name="apple_geom" type="sphere" size=".025" mass=".1"/></body>
      <body name="object_board" pos=".5 0 .02"><geom name="board_geom" type="box" size=".08 .08 .02"/></body>
      <body name="object_drawer_shell" pos=".2 .1 .05" quat=".9238795325 0 0 .3826834324">
        <geom name="drawer_shell_geom" type="box" size=".10 .08 .04"/>
        <body name="object_drawer" pos="0 -.05 0">
          <joint name="drawer_slide" type="slide" axis="0 -1 0" range="0 .2"/>
          <geom name="drawer_geom" type="box" size=".08 .06 .03"/>
        </body>
      </body>
      <camera name="main" pos="0 -1 .8" xyaxes="1 0 0 0 .6 .8"
              resolution="320 180" sensorsize="1 .5625" focal=".5 .5" principal=".1 -.05"/>
    </worldbody><actuator>{motors}</actuator></mujoco>"""
    return mujoco.MjModel.from_xml_string(xml)


def _world(*, render=False, width=640, height=360, initial_translation=None,
           cup_release_control="legacy_object_distance"):
    trajectory = {
        "timestamps": np.array([0., .01]),
        "joints": np.zeros((2, 2, 6)),
        "gripper": np.ones((2, 2)),
    }
    record = {"objects": [
        {"name": "apple", "asset": "cube"},
        {"name": "board", "asset": "board"},
        {"name": "drawer_shell", "asset": "drawer_shell"},
        {"name": "drawer", "asset": "drawer"},
    ]}
    return TwinWorld(_mini_model(), record, trajectory, render=render, width=width, height=height,
                     cup_release_control=cup_release_control,
                     initial_translation=({"apple": [.01, 0, 0]}
                                          if initial_translation is None else initial_translation))


def test_portable_release_pause_uses_only_tcp_displacement_or_timeout():
    world = _world(cup_release_control="portable_tcp_displacement_v1")
    try:
        world.initialize(np.zeros((2, 6)), np.zeros(2))
        apple = world.model.body("object_apple").id
        world.object_specs[apple]["asset"] = "025_mug"
        world._held_by_arm["left"] = apple
        world._previous_gripper_command[0] = 0.0
        q = np.zeros((2, 6)); g = np.array([1.0, 0.0]); v = np.zeros((2, 6))
        world._servo(q, g, v)
        assert "left" in world._release_pause
        # Moving hidden object truth cannot end portable release pause.
        free_joint = int(world.model.body("object_apple").jntadr[0])
        world.data.qpos[int(world.model.jnt_qposadr[free_joint])] += 1.0
        mujoco.mj_forward(world.model, world.data)
        world._servo(q, g, v)
        assert "left" in world._release_pause
        object_id, started, aperture, release_tcp = world._release_pause["left"]
        world._release_pause["left"] = (object_id, started - 1.01, aperture, release_tcp)
        world._servo(q, g, v)
        assert "left" not in world._release_pause
    finally:
        world.close()


def test_native_badq_and_time_regression_fail_closed_even_after_finite_reset():
    world = _world()
    try:
        world.initialize(np.zeros((2, 6)), np.ones(2))
        world._warning_callback(
            "Nan, Inf or huge value in QACC at DOF 26. The simulation is unstable. Time = 0.2290.")
        safety = world.numerical_safety()
        assert not safety["simulation_finite"] and not safety["passed"]
        assert safety["numerical_diagnostics"][0]["kind"] == "native_mujoco_warning"
        world._numerical_integrity_ok = True
        world._record_numerical_step(float(world.data.time) + .01)
        safety = world.numerical_safety()
        assert not safety["simulation_finite"]
        assert safety["numerical_diagnostics"][-1]["kind"] == "physical_time_regression"
    finally:
        world.close()


def test_array_contract_copies_and_rejects_nonfinite():
    source = np.arange(12, dtype=float).reshape(2, 6)
    result = _array(source, (2, 6), "joints")
    source[:] = -1
    assert result[0, 0] == 0
    with pytest.raises(ValueError, match="finite"):
        _array([[np.nan]], (1, 1), "bad")


def test_mapping_trajectory_is_copied():
    source = {"timestamps": np.array([0., 1.]), "gripper": np.ones((2, 2))}
    loaded = _load_npz(source)
    source["timestamps"][0] = 5
    assert loaded["timestamps"][0] == 0


def test_force_step_initializes_once_and_exposes_robot_only_runtime_data():
    world = _world()
    try:
        proprio = world.initialize(np.zeros((2, 6)), np.ones(2))
        assert proprio.tcp_position.shape == (2, 3)
        assert proprio.tcp_rotation.shape == (2, 3, 3)
        assert proprio.joint_pos.shape == (2, 6)
        assert proprio.finger_qpos.shape == (2, 2)
        with pytest.raises(RuntimeError, match="only once"):
            world.initialize(np.zeros((2, 6)), np.ones(2))
        step = world.step_command(np.zeros((2, 6)), np.ones(2),
                                  duration_sec=.003, observe=False)
        assert step.physics_steps == 3
        assert step.physical_time_sec == pytest.approx(.003)
        assert isinstance(step.contact_evidence["roles"]["apple"]["held_by"]["left"], bool)
        assert "peak_lift_mm" not in step.contact_evidence["roles"]["apple"]
        final = world.finalize_evaluator()
        assert final["state_writes_after_initialization"] == 0
        assert not final["object_external_forces"]
        assert final["grasp_welds"] == 0
    finally:
        world.close()


def test_scratch_kinematics_and_visible_facts_do_not_mutate_state():
    world = _world()
    try:
        world.initialize(np.zeros((2, 6)), np.ones(2))
        before = world.data.qpos.copy()
        J = world.jacobian("left", np.full(6, .01))
        solved = world.solve_ik_delta("left", np.array([.001, 0, 0, 0, 0, 0]))
        assert J.shape == (6, 6)
        assert solved.shape == (6,)
        np.testing.assert_array_equal(world.data.qpos, before)

        def frame(z):
            role = RoleObservation("apple", True, None, np.array([-.3, 0, z]), .01, 20, (0, 0, 4, 4), np.ones((1, 1), dtype=bool))
            return VisibleRGBDFrame(0., 0., np.zeros((1, 1, 3), np.uint8),
                                    np.ones((1, 1)), np.eye(3), np.eye(4), {"apple": role})

        initial = world.visible_stage_facts("apple", frame(.10), arm="left")
        world._current_loaded["object_apple"]["left"] = True
        world._current_object_contacts.add(frozenset((
            world.model.body("object_apple").id,
            world.model.body("object_board").id,
        )))
        contact = world.visible_stage_facts(
            "apple", frame(.10), arm="left", support="board")
        world._current_object_table_contacts.add(world.model.body("object_apple").id)
        table_contact = world.visible_stage_facts("apple", frame(.10), arm="left")
        lifted = world.visible_stage_facts("apple", frame(.15), arm="left")
        assert not initial["lifted"]
        assert contact["single_finger_loaded_contact"]
        assert contact["object_support_contact"]
        assert table_contact["object_table_support_contact"]
        assert world.contact_evidence()["roles"]["apple"]["object_table_support_contact"]
        assert lifted["lifted"]
        assert all(isinstance(value, bool) for value in lifted.values())
    finally:
        world.close()


def test_initial_translation_preserves_attached_fixture_world_semantics_without_step():
    baseline = _world(initial_translation={})
    common = np.array([.017, -.009, .004])
    moved = _world(initial_translation={"drawer_shell": common, "drawer": common})
    child_delta = np.array([-.006, .013, .002])
    free_delta = np.array([.004, -.003, .001])
    split = _world(initial_translation={
        "drawer_shell": common, "drawer": child_delta, "apple": free_delta})
    try:
        shell0 = baseline.data.body("object_drawer_shell").xpos.copy()
        drawer0 = baseline.data.body("object_drawer").xpos.copy()
        np.testing.assert_allclose(
            moved.data.body("object_drawer_shell").xpos - shell0, common, atol=1e-12)
        np.testing.assert_allclose(
            moved.data.body("object_drawer").xpos - drawer0, common, atol=1e-12)
        np.testing.assert_allclose(
            moved.data.body("object_drawer").xpos - moved.data.body("object_drawer_shell").xpos,
            drawer0 - shell0, atol=1e-12)
        np.testing.assert_allclose(
            split.data.body("object_drawer_shell").xpos - shell0, common, atol=1e-12)
        np.testing.assert_allclose(
            split.data.body("object_drawer").xpos - drawer0, child_delta, atol=1e-12)
        np.testing.assert_allclose(
            split.data.body("object_apple").xpos - baseline.data.body("object_apple").xpos,
            free_delta, atol=1e-12)
        np.testing.assert_allclose(
            split._requested_initial_translation_world["drawer_shell"], common)
        np.testing.assert_allclose(
            split._requested_initial_translation_world["drawer"], child_delta)
        assert baseline.data.time == moved.data.time == split.data.time == 0.
        assert baseline._state_writes_after_initialization == 0
        assert moved._state_writes_after_initialization == 0
        assert split._state_writes_after_initialization == 0
    finally:
        baseline.close(); moved.close(); split.close()


def test_sparse_segmentation_edge_is_missing_not_a_false_centroid():
    world = _world(render=True, width=160, height=90)
    try:
        world.initialize(np.zeros((2, 6)), np.ones(2))
        geom_id = next(iter(world.role_geom_ids["apple"]))
        depth = np.ones((3, 4), dtype=np.float32)
        segmentation = np.full((3, 4, 2), -1, dtype=np.int32)
        segmentation[0, 0] = [geom_id, int(mujoco.mjtObj.mjOBJ_GEOM)]
        world.observer._render = lambda: (
            np.zeros((3, 4, 3), dtype=np.uint8), depth, segmentation)
        world.observer._camera = lambda: (np.eye(3), np.eye(4))
        sparse = world.observe_visible(["apple"]).roles["apple"]
        assert not sparse.present
        assert sparse.missing_reason == "insufficient_visible_depth"
        assert sparse.pixel_count == 1
        assert sparse.centroid_world_m is None

        segmentation[:] = [geom_id, int(mujoco.mjtObj.mjOBJ_GEOM)]
        depth[0, 0] = 100.  # A silhouette depth mismatch must not dominate.
        visible = world.observe_visible(["apple"]).roles["apple"]
        assert visible.present and visible.pixel_count == 12
        assert visible.centroid_world_m[2] == pytest.approx(1.)
    finally:
        world.close()


def test_private_stage_goal_returns_only_booleans_and_debounces():
    world = _world()
    try:
        world.initialize(np.zeros((2, 6)), np.ones(2))
        apple = world.model.body("object_apple").id
        board = world.model.body("object_board").id
        world._ever_loaded.add(("object_apple", "left"))
        world._current_object_contacts.add(frozenset((apple, board)))

        first = world.stage_goal_facts(
            "apple", "board", arm="left", goal_relation="supported",
            debounce_sec=.1)
        assert first["goal_supported"] is False
        assert first["supported"] is False
        assert first["stage_goal_complete"] is False
        world.data.time = .1
        settled = world.stage_goal_facts(
            "apple", "board", arm="left", goal_relation="supported",
            debounce_sec=.1)
        assert settled["goal_supported"] is True
        assert all(value is None or isinstance(value, bool) for value in settled.values())

        joint_id = int(world.model.body_jntadr[apple])
        address = int(world.model.jnt_qposadr[joint_id])
        world.data.qpos[address + 2] += .05
        mujoco.mj_forward(world.model, world.data)
        lifted = world.stage_goal_facts(
            "apple", "symbolic_lift_threshold", arm="left",
            goal_relation="lifted")
        assert lifted["goal_lifted"] is True
        world._current_object_table_contacts.add(apple)
        handover = world.stage_goal_facts(
            "apple", "table_handover_region", arm="left",
            goal_relation="accessible_to_right_arm")
        assert handover["goal_handover"] is True
        assert handover["accessible_to_right_arm"] is True
        assert all(value is None or isinstance(value, bool)
                   for value in (*lifted.values(), *handover.values()))
        with pytest.raises(KeyError, match="unknown semantic role"):
            world.stage_goal_facts(
                "apple", "symbolic_invalid_target", arm="left",
                goal_relation="supported")
        with pytest.raises(ValueError, match="between 0 and .15"):
            world.stage_goal_facts(
                "apple", "board", arm="left", goal_relation="lifted",
                debounce_sec=.151)
    finally:
        world.close()


def test_visible_rgbd_uses_scene_calibration_and_mask_depth_geometry():
    """Evaluator truth appears only here as an offline observer diagnostic."""
    world = _world(render=True, width=320, height=180)
    try:
        world.initialize(np.zeros((2, 6)), np.ones(2))
        frame = world.observe_visible(["apple"])
        np.testing.assert_allclose(
            frame.camera_K,
            np.array([[160., 0., 128.], [0., 160., 74.], [0., 0., 1.]]),
            atol=1e-6,
        )
        role = frame.roles["apple"]
        assert role.present and role.mask.shape == (180, 320) and role.mask.dtype == bool
        evaluator_truth = world.data.body("object_apple").xpos.copy()
        assert np.linalg.norm(role.centroid_world_m - evaluator_truth) < .04
        assert frame.camera_convention == "opencv_x_right_y_down_z_forward"
    finally:
        world.close()
