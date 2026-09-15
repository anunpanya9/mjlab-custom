"""Tests for the G1 VLA playground env (g1_vla)."""

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.manipulation.config.g1_vla.env_cfgs import (
  ARM_GRIPPER_JOINTS,
  g1_vla_env_cfg,
)


def _build_env() -> ManagerBasedRlEnv:
  cfg = g1_vla_env_cfg()
  cfg.scene.num_envs = 1
  return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def test_action_space_is_16_dof():
  """The VLA outputs 16 values (14 arm + 2 gripper); the env must accept exactly
  that many, so legs/waist/ankles are excluded from the action."""
  env = _build_env()
  assert env.action_manager.total_action_dim == 16


def test_action_targets_are_arm_and_gripper_only():
  """Every action target must be one of the 16 arm/gripper joints (no legs)."""
  env = _build_env()
  targets = set(env.action_manager.get_term("joint_pos")._target_names)
  assert targets == set(ARM_GRIPPER_JOINTS)


def test_three_dataset_cameras_present():
  """The three dataset camera views must exist so the policy sees the inputs it
  was trained on."""
  cfg = g1_vla_env_cfg()
  names = {s.name for s in (cfg.scene.sensors or ())}
  assert names == {"cam_left_high", "cam_left_wrist", "cam_right_wrist"}


def test_env_steps_with_16dim_action():
  """A 16-dim zero action steps the env without error."""
  env = _build_env()
  env.reset()
  env.step(torch.zeros(1, 16))
