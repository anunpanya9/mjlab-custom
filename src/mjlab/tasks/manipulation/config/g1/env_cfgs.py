"""Stationary G1 lift-cube manipulation task.

The G1 is welded at the pelvis (fixed-base) so only the arms and Dex3-1 hands
move. A cube rests on a table in front of the robot; the task is to bring the
right hand's grasp site to the cube and lift it to a commanded height. This is
the Phase-1 grasp milestone on the road to loco-manipulation and, eventually, a
VLA (see docs/development/g1_manipulation_vla.md).

The MDP is cloned from the YAM lift-cube task; the differences are the robot,
the added table, the reach-height of the object pose range, and the grasp
site / fingertip geoms, which are G1-specific.
"""

from typing import Any, Literal

import mujoco

from mjlab.asset_zoo.robots import (
  G1_WITH_HANDS_ACTION_SCALE,
  get_g1_with_hands_fixed_base_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import CameraSensorCfg, ContactSensorCfg
from mjlab.tasks.manipulation import mdp as manipulation_mdp
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg

# Table top height and the cube's edge, used to place the cube on the table and
# to set the reachable object pose range in front of the robot.
_TABLE_HEIGHT = 0.8
_CUBE_SIZE = 0.02

# The right-hand grasp site and its fingertip collision geoms. Phase 1 uses a
# single hand; bimanual comes later.
_GRASP_SITE = "right_grasp_site"
_FINGERTIP_GEOMS = r"right_hand_(thumb_2|middle_1|index_1)_collision"
# End-effector body whose contact with the ground/table is a failure.
_EE_BODY = "right_wrist_yaw_link"

# Head camera: mounted on the torso, tilted down to frame the table workspace.
# The pose was tuned by rendering (scratchpad/test_head_cam3.py): from
# pos=(0.1, 0, 0.12) in the torso frame, this quaternion looks forward and down
# at the table center, seeing both the cube and the hands. Fixed-base keeps the
# torso still, so a static quaternion suffices (Phase 3 will revisit this once
# the base moves).
_HEAD_CAM_PARENT = "robot/torso_link"
_HEAD_CAM_POS = (0.1, 0.0, 0.12)
_HEAD_CAM_QUAT = (-0.4922, -0.332, 0.4499, 0.6671)
_HEAD_CAM_FOVY = 75.0


def get_cube_spec(
  cube_size: float = _CUBE_SIZE,
  mass: float = 0.05,
  rgba: tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0),
) -> mujoco.MjSpec:
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="cube")
  body.add_freejoint(name="cube_joint")
  body.add_geom(
    name="cube_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(cube_size,) * 3,
    mass=mass,
    rgba=rgba,
  )
  return spec


def get_table_spec(
  height: float = _TABLE_HEIGHT,
  half_x: float = 0.3,
  half_y: float = 0.4,
) -> mujoco.MjSpec:
  """A static table: a box top welded to the world at the given height."""
  spec = mujoco.MjSpec()
  body = spec.worldbody.add_body(name="table", pos=(0.45, 0.0, height / 2.0))
  body.add_geom(
    name="table_collision",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(half_x, half_y, height / 2.0),
    rgba=(0.6, 0.5, 0.4, 1.0),
  )
  return spec


def g1_lift_cube_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": get_g1_with_hands_fixed_base_robot_cfg(),
    "table": EntityCfg(spec_fn=get_table_spec),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  # Position the robot's mocap base at each env origin.
  # (reset_base already handles this via reset_root_state_uniform.)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_WITH_HANDS_ACTION_SCALE

  # Wire the G1 right-hand grasp site into the reach observation and reward.
  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    _GRASP_SITE,
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = (_GRASP_SITE,)

  # Fingertip friction randomization on the Dex3 fingertips.
  for term in (
    "fingertip_friction_slide",
    "fingertip_friction_spin",
    "fingertip_friction_roll",
  ):
    cfg.events[term].params["asset_cfg"].geom_names = _FINGERTIP_GEOMS

  # Sample the cube on the table top, within the right arm's reach.
  cube_z = _TABLE_HEIGHT + _CUBE_SIZE
  command = cfg.commands["lift_height"]
  assert isinstance(command, LiftingCommandCfg)
  command.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=(0.35, 0.55),
    y=(-0.3, 0.0),
    z=(cube_z, cube_z + 0.02),
    yaw=(-3.14, 3.14),
  )

  # End-effector to ground/table collision sensor targets the right wrist.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = _EE_BODY

  # More contacts now that fingers and a table are in the scene.
  cfg.sim.nconmax = max(cfg.sim.nconmax or 55, 200)

  cfg.viewer.body_name = "torso_link"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (4.0, 4.0)

  return cfg


def g1_lift_cube_vision_env_cfg(
  cam_type: Literal["rgb", "depth"] = "rgb",
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Vision variant: the actor sees the object through the head camera instead
  of the privileged cube pose. The critic keeps the privileged state
  (asymmetric actor-critic), which stabilizes PPO while the actor learns from
  pixels. This is the Phase-2 milestone (see the dev doc)."""
  cfg = g1_lift_cube_env_cfg(play=play)

  # Add the torso head camera.
  cam_cfg = CameraSensorCfg(
    name="head_cam",
    parent_body=_HEAD_CAM_PARENT,
    pos=_HEAD_CAM_POS,
    quat=_HEAD_CAM_QUAT,
    fovy=_HEAD_CAM_FOVY,
    width=64,
    height=64,
    data_types=(cam_type,),
    enabled_geom_groups=(0, 3),
    use_shadows=False,
    use_textures=True,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (cam_cfg,)

  # Camera observation group (separate, not concatenated with the state obs).
  param_kwargs: dict[str, Any] = {"sensor_name": "head_cam"}
  if cam_type == "depth":
    param_kwargs["cutoff_distance"] = 1.5
    cam_func = manipulation_mdp.camera_depth
  else:
    cam_func = manipulation_mdp.camera_rgb
  cfg.observations["camera"] = ObservationGroupCfg(
    terms={f"head_{cam_type}": ObservationTermCfg(func=cam_func, params=param_kwargs)},
    enable_corruption=False,
    concatenate_terms=True,
  )

  # Randomize cube color so an RGB policy cannot memorize a fixed appearance.
  if cam_type == "rgb":
    cfg.events["cube_color"] = EventTermCfg(
      func=dr.geom_rgba,
      mode="reset",
      params={
        "asset_cfg": SceneEntityCfg("cube", geom_names=(".*",)),
        "operation": "abs",
        "distribution": "uniform",
        "axes": [0, 1, 2],
        "ranges": (0.0, 1.0),
      },
    )

  # Drop the privileged cube observations from the actor; the critic keeps them.
  actor_obs = cfg.observations["actor"]
  actor_obs.terms.pop("ee_to_cube", None)
  actor_obs.terms.pop("cube_to_goal", None)

  # Give the actor the commanded goal position in the hand frame (non-privileged
  # since it is the task command, not the object state).
  actor_obs.terms["goal_position"] = ObservationTermCfg(
    func=manipulation_mdp.target_position,
    params={
      "command_name": "lift_height",
      "asset_cfg": SceneEntityCfg("robot", site_names=(_GRASP_SITE,)),
    },
  )

  return cfg
