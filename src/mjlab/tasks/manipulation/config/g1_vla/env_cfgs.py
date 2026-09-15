"""G1 pick-and-place scene for running a Dex1 VLA policy in mjlab.

This is not an RL task — it exists to give a fine-tuned GR00T VLA (trained on the
Unitree ``G1_Dex1_PickPlaceRedBlock`` dataset) a place to act in sim. The robot
is the parallel-jaw gripper G1 whose 16-DOF action space matches the dataset, the
scene is a table with a red cube and a yellow box (mirroring the dataset's
"place the red block into the yellow box" task), and three RGB cameras reproduce
the dataset's views (a head/front camera plus one on each wrist) so the policy
sees observations shaped the way it was trained on.

The VLA bridge (``vla_bridge/``) renders these cameras + reads joint state, sends
them to a GR00T policy server, and applies the returned action chunk.
"""

from __future__ import annotations

from mjlab.asset_zoo.robots.unitree_g1.g1_gripper_constants import (
  get_g1_with_grippers_fixed_base_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.sensor import CameraSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.config.g1.env_cfgs import (
  _TABLE_HEIGHT,
  get_cube_spec,
  get_table_spec,
)
from mjlab.tasks.manipulation.config.g1_place_box.env_cfgs import get_box_spec
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg

# Cameras reproduce the dataset's three views. The head camera reuses the tuned
# torso-mounted pose from the lift-cube vision task; the wrist cameras sit just
# past each wrist, looking down the arm toward the grasp.
_HEAD_CAM_PARENT = "robot/torso_link"
_HEAD_CAM_POS = (0.1, 0.0, 0.12)
_HEAD_CAM_QUAT = (-0.4922, -0.332, 0.4499, 0.6671)
_HEAD_CAM_FOVY = 75.0

_WRIST_CAM_POS = (0.03, 0.0, 0.04)
# Look forward down the wrist toward the fingertips (identity-ish, tilted down).
_WRIST_CAM_QUAT = (0.5, -0.5, 0.5, -0.5)
_WRIST_CAM_FOVY = 80.0

# The VLA was trained on 224x224 RGB frames.
_CAM_RES = 224

# The 16 joints the VLA drives, in the dataset's state/action order:
# left arm (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw), right arm
# (same), then left gripper, right gripper.
#
# NOTE ON ORDER: mjlab groups the resolved action targets by side, so the env's
# actual action order comes out as [left arm 7, LEFT GRIPPER, right arm 7, RIGHT
# GRIPPER], not [left arm 7, right arm 7, left gripper, right gripper]. The two
# gripper scalars therefore sit at env indices 7 and 15, while the dataset puts
# them at 14 and 15. The VLA bridge MUST remap the policy's 16-vector into the
# env's target order before applying it (and remap joint state the other way when
# building the observation). ``env.action_manager.get_term("joint_pos")._target_names``
# gives the authoritative env order to remap against.
ARM_GRIPPER_JOINTS: tuple[str, ...] = (
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
  "left_gripper_joint",
  "right_gripper_joint",
)


def _rgb_camera(name: str, parent: str, pos, quat, fovy) -> CameraSensorCfg:
  return CameraSensorCfg(
    name=name,
    parent_body=parent,
    pos=pos,
    quat=quat,
    fovy=fovy,
    width=_CAM_RES,
    height=_CAM_RES,
    data_types=("rgb",),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )


def g1_vla_env_cfg() -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": get_g1_with_grippers_fixed_base_robot_cfg(),
    "table": EntityCfg(spec_fn=get_table_spec),
    "box": EntityCfg(spec_fn=get_box_spec),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  # The lift-cube base drags in Dex3-specific rewards, terminations, events and a
  # contact sensor that reference hand sites/geoms this gripper robot does not
  # have. This env is a VLA playground, not an RL task, so strip that stack and
  # keep only the scene, the joint-position action, and (added below) cameras.
  cfg.scene.sensors = ()
  cfg.rewards = {}
  cfg.terminations = {"time_out": cfg.terminations["time_out"]}
  cfg.curriculum = {}
  cfg.events = {
    k: v for k, v in cfg.events.items() if k in ("reset_base", "reset_robot_joints")
  }
  # Drop the RL actor/critic observation groups entirely. The VLA reads joint
  # state directly from the entity (via the bridge), and its vision comes from the
  # camera group added below — it does not consume these privileged RL obs.
  cfg.observations = {}

  # Restrict the action to exactly the 16 joints the VLA drives, IN THE DATASET'S
  # ORDER (left arm 7, right arm 7, left gripper, right gripper). preserve_order
  # makes action index i map to ARM_GRIPPER_JOINTS[i], so the policy's 16-vector
  # lines up one-to-one with the sim joints. Legs/waist/ankles are welded
  # (fixed-base) and left unactuated by the action.
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = ARM_GRIPPER_JOINTS
  joint_pos_action.preserve_order = True
  joint_pos_action.scale = 0.5

  # Three RGB cameras matching the dataset's views.
  head = _rgb_camera(
    "cam_left_high", _HEAD_CAM_PARENT, _HEAD_CAM_POS, _HEAD_CAM_QUAT, _HEAD_CAM_FOVY
  )
  left_wrist = _rgb_camera(
    "cam_left_wrist",
    "robot/left_wrist_yaw_link",
    _WRIST_CAM_POS,
    _WRIST_CAM_QUAT,
    _WRIST_CAM_FOVY,
  )
  right_wrist = _rgb_camera(
    "cam_right_wrist",
    "robot/right_wrist_yaw_link",
    _WRIST_CAM_POS,
    _WRIST_CAM_QUAT,
    _WRIST_CAM_FOVY,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (head, left_wrist, right_wrist)

  # Expose the three camera images as observation terms so the bridge can read
  # them straight off the env. (State/language are supplied by the bridge itself.)
  cam_terms = {
    "cam_left_high": ObservationTermCfg(
      func=manipulation_mdp.camera_rgb, params={"sensor_name": "cam_left_high"}
    ),
    "cam_left_wrist": ObservationTermCfg(
      func=manipulation_mdp.camera_rgb, params={"sensor_name": "cam_left_wrist"}
    ),
    "cam_right_wrist": ObservationTermCfg(
      func=manipulation_mdp.camera_rgb, params={"sensor_name": "cam_right_wrist"}
    ),
  }
  cfg.observations["cameras"] = ObservationGroupCfg(
    terms=cam_terms, enable_corruption=False, concatenate_terms=False
  )

  # Place the cube on the table in front of the robot (the box, added as a static
  # entity, sits to the side), matching the dataset layout (block -> yellow box).
  cube_z = _TABLE_HEIGHT + 0.02
  command = cfg.commands["lift_height"]
  assert isinstance(command, LiftingCommandCfg)
  command.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=(0.35, 0.5), y=(-0.25, -0.05), z=(cube_z, cube_z + 0.02), yaw=(-3.14, 3.14)
  )

  cfg.viewer.body_name = "torso_link"
  cfg.episode_length_s = 30.0
  return cfg
