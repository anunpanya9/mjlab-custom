"""Tests for g1_constants.py."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.unitree_g1 import g1_constants
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr


@pytest.fixture(scope="module")
def g1_entity() -> Entity:
  return Entity(g1_constants.get_g1_robot_cfg())


@pytest.fixture(scope="module")
def g1_model(g1_entity: Entity) -> mujoco.MjModel:
  return g1_entity.spec.compile()


# fmt: off
@pytest.mark.parametrize(
  "actuator_config,stiffness,damping",
  [
    (g1_constants.G1_ACTUATOR_5020, g1_constants.STIFFNESS_5020, g1_constants.DAMPING_5020),
    (g1_constants.G1_ACTUATOR_7520_14, g1_constants.STIFFNESS_7520_14, g1_constants.DAMPING_7520_14),
    (g1_constants.G1_ACTUATOR_7520_22, g1_constants.STIFFNESS_7520_22, g1_constants.DAMPING_7520_22),
    (g1_constants.G1_ACTUATOR_4010, g1_constants.STIFFNESS_4010, g1_constants.DAMPING_4010),
    (g1_constants.G1_ACTUATOR_WAIST, g1_constants.STIFFNESS_5020 * 2, g1_constants.DAMPING_5020 * 2),
    (g1_constants.G1_ACTUATOR_ANKLE, g1_constants.STIFFNESS_5020 * 2, g1_constants.DAMPING_5020 * 2),
  ],
)
# fmt: on
def test_actuator_parameters(g1_model, actuator_config, stiffness, damping):
  for i in range(g1_model.nu):
    actuator = g1_model.actuator(i)
    actuator_name = actuator.name
    matches = any(
      re.match(pattern, actuator_name) for pattern in actuator_config.target_names_expr
    )
    if matches:
      assert actuator.gainprm[0] == stiffness
      assert actuator.biasprm[1] == -stiffness
      assert actuator.biasprm[2] == -damping
      assert actuator.forcerange[0] == -actuator_config.effort_limit
      assert actuator.forcerange[1] == actuator_config.effort_limit


def test_keyframe_base_position(g1_model) -> None:
  data = mujoco.MjData(g1_model)
  mujoco.mj_resetDataKeyframe(g1_model, data, 0)
  mujoco.mj_forward(g1_model, data)
  np.testing.assert_array_equal(data.qpos[:3], g1_constants.KNEES_BENT_KEYFRAME.pos)
  np.testing.assert_array_equal(data.qpos[3:7], g1_constants.KNEES_BENT_KEYFRAME.rot)


def test_keyframe_joint_positions(g1_entity, g1_model) -> None:
  """Test that keyframe joint positions match the configuration."""
  key = g1_model.key("init_state")
  expected_joint_pos = g1_constants.KNEES_BENT_KEYFRAME.joint_pos
  assert expected_joint_pos is not None
  expected_values = resolve_expr(expected_joint_pos, g1_entity.joint_names, 0.0)
  for joint_name, expected_value in zip(
    g1_entity.joint_names, expected_values, strict=True
  ):
    joint = g1_model.joint(joint_name)
    qpos_idx = joint.qposadr[0]
    actual_value = key.qpos[qpos_idx]
    np.testing.assert_allclose(
      actual_value,
      expected_value,
      rtol=1e-5,
      err_msg=f"Joint {joint_name} position mismatch: "
      f"expected {expected_value}, got {actual_value}",
    )


def test_foot_collision_geoms(g1_model) -> None:
  foot_pattern = r"^(left|right)_foot[1-7]_collision$"
  for i in range(g1_model.ngeom):
    geom = g1_model.geom(i)
    geom_name = geom.name
    # Foot collision geoms should have condim=3, priority=1, and friction=0.6.
    if re.match(foot_pattern, geom_name):
      assert geom.condim == 3
      assert geom.priority == 1
      assert geom.friction[0] == 0.6


def test_non_foot_collision_geoms(g1_model) -> None:
  foot_pattern = r"^(left|right)_foot[1-7]_collision$"
  for i in range(g1_model.ngeom):
    geom = g1_model.geom(i)
    geom_name = geom.name
    if "_collision" not in geom_name:
      continue
    # Non-foot collision geoms should have condim=1
    if not re.match(foot_pattern, geom_name):
      assert geom.condim == 1


def test_collision_geom_count(g1_model) -> None:
  # There should be 7 geoms (capsules) per foot.
  collision_geoms = [
    g1_model.geom(i).name
    for i in range(g1_model.ngeom)
    if "_collision" in g1_model.geom(i).name
  ]
  assert len(collision_geoms) > 0
  foot_geoms = [
    name
    for name in collision_geoms
    if re.match(r"^(left|right)_foot[1-7]_collision$", name)
  ]
  assert len(foot_geoms) == 14


def test_g1_entity_creation(g1_entity) -> None:
  assert g1_entity.num_actuators == 29
  assert g1_entity.num_joints == 29
  assert g1_entity.is_actuated
  assert not g1_entity.is_fixed_base


def test_g1_actuators_configured_correctly(g1_model):
  """Verify that all G1 actuators have correct control and force limiting.

  All 29 G1 actuators should have ctrllimited=False (allowing setpoints beyond
  joint limits) and forcelimited=True (limiting forces to effort limits).
  """
  for i in range(g1_model.nu):
    actuator_name = g1_model.actuator(i).name
    assert g1_model.actuator_ctrllimited[i] == 0, (
      f"Actuator '{actuator_name}' has ctrllimited=True, expected False"
    )
    assert g1_model.actuator_forcelimited[i] == 1, (
      f"Actuator '{actuator_name}' has forcelimited=False, expected True"
    )


##
# G1 with Dex3-1 hands.
##


@pytest.fixture(scope="module")
def g1_hands_entity() -> Entity:
  return Entity(g1_constants.get_g1_with_hands_robot_cfg())


@pytest.fixture(scope="module")
def g1_hands_model(g1_hands_entity: Entity) -> mujoco.MjModel:
  return g1_hands_entity.spec.compile()


def test_g1_with_hands_dof_count(g1_hands_entity) -> None:
  # 29 body DOF + 7 per hand * 2 hands = 43.
  assert g1_hands_entity.num_joints == 43
  assert g1_hands_entity.num_actuators == 43
  assert not g1_hands_entity.is_fixed_base


def test_g1_with_hands_finger_joints_actuated(g1_hands_model) -> None:
  """Every Dex3-1 finger joint must be actuated with the hand effort limits."""
  finger_efforts = {
    "hand_thumb_0": 2.45,
    "hand_thumb_1": 1.4,
    "hand_thumb_2": 1.4,
    "hand_middle_0": 1.4,
    "hand_middle_1": 1.4,
    "hand_index_0": 1.4,
    "hand_index_1": 1.4,
  }
  for side in ("left", "right"):
    for stem, effort in finger_efforts.items():
      name = f"{side}_{stem}_joint"
      actuator = g1_hands_model.actuator(name)
      np.testing.assert_allclose(actuator.forcerange[1], effort, rtol=1e-6)


def test_g1_with_hands_grasp_sites(g1_hands_model) -> None:
  for side in ("left", "right"):
    site = g1_hands_model.site(f"{side}_grasp_site")
    assert site.id >= 0


def test_g1_with_hands_finger_collision_geoms_named(g1_hands_model) -> None:
  """Hand collision geoms must be named so CollisionCfg (`.*_collision`)
  applies condim=1 to them like the rest of the body."""
  hand_collisions = [
    g1_hands_model.geom(i).name
    for i in range(g1_hands_model.ngeom)
    if "hand" in g1_hands_model.geom(i).name
    and g1_hands_model.geom(i).name.endswith("_collision")
  ]
  # palm + 7 finger links per hand * 2 hands = 16.
  assert len(hand_collisions) == 16
  for i in range(g1_hands_model.ngeom):
    geom = g1_hands_model.geom(i)
    if geom.name in hand_collisions:
      assert geom.condim == 1
