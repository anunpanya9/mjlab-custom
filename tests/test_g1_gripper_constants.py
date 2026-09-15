"""Tests for g1_gripper_constants.py (the Dex1-style parallel-jaw G1)."""

import mujoco
import pytest

from mjlab.asset_zoo.robots.unitree_g1 import g1_gripper_constants
from mjlab.entity import Entity


@pytest.fixture(scope="module")
def gripper_entity() -> Entity:
  return Entity(g1_gripper_constants.get_g1_with_grippers_fixed_base_robot_cfg())


@pytest.fixture(scope="module")
def gripper_model(gripper_entity: Entity) -> mujoco.MjModel:
  return gripper_entity.spec.compile()


def _joint_names(model: mujoco.MjModel) -> list[str]:
  return [model.joint(i).name for i in range(model.njnt)]


def test_arm_joints_present(gripper_model: mujoco.MjModel):
  """The 14 arm joints must survive (the VLA drives them directly)."""
  names = _joint_names(gripper_model)
  arm = [n for n in names if any(x in n for x in ("shoulder", "elbow", "wrist"))]
  assert len(arm) == 14


def test_one_driving_gripper_joint_per_hand(gripper_model: mujoco.MjModel):
  """Each hand exposes exactly one driving gripper joint, so the action space is
  14 arm + 2 gripper = 16, matching the Dex1 teleop dataset."""
  names = _joint_names(gripper_model)
  drive = [n for n in names if n in ("left_gripper_joint", "right_gripper_joint")]
  assert sorted(drive) == ["left_gripper_joint", "right_gripper_joint"]


def test_mirror_fingers_coupled(gripper_model: mujoco.MjModel):
  """The second finger of each hand follows via a joint equality constraint, so
  it must not add its own actuated DOF but must exist and be coupled."""
  names = _joint_names(gripper_model)
  mirror = [n for n in names if "gripper_mirror_joint" in n]
  assert len(mirror) == 2
  # two equality constraints, one coupling each hand's mirror to its drive
  assert gripper_model.neq == 2


def test_gripper_actuators_do_not_target_mirror(gripper_entity: Entity):
  """Only the driving joints are actuated; the mirror joints follow the
  constraint, not the action."""
  targets = g1_gripper_constants.G1_ACTUATOR_GRIPPER.target_names_expr
  assert "left_gripper_joint" in targets
  assert "right_gripper_joint" in targets
  assert all("mirror" not in t for t in targets)
