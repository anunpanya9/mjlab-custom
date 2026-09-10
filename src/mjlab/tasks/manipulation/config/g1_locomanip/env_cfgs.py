"""G1 loco-manipulation: walk to an object, pick it up, carry it to a goal.

This is the Phase-3 milestone (see docs/development/g1_manipulation_vla.md): the
G1 is floating-base (it can walk) with two Dex3-1 hands, and must locomote to a
cube placed a couple of meters away, grasp it, and carry it to a commanded goal
location. It reuses the lift-cube manipulation stack (LiftingCommand, staged
grasp rewards, fingertip friction) but with the object and goal placed far
enough that the robot must walk rather than just reach.

Whole-body loco-manipulation is the hard research crux of this project: the
reward must balance stable locomotion against manipulation precision. This
config establishes a correct, buildable task; the reward weights here are a
starting point meant to be tuned during GPU training.
"""

from mjlab.asset_zoo.robots import (
  G1_WITH_HANDS_ACTION_SCALE,
  get_g1_with_hands_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.manipulation.config.g1.env_cfgs import get_cube_spec
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg

# Right hand grasp site + fingertip collision geoms (single hand for now).
_GRASP_SITE = "right_grasp_site"
_FINGERTIP_GEOMS = r"right_hand_(thumb_2|middle_1|index_1)_collision"
_EE_BODY = "right_wrist_yaw_link"


def g1_locomanip_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()

  # Floating-base G1 with Dex3 hands: it can walk AND manipulate.
  cfg.scene.entities = {
    "robot": get_g1_with_hands_robot_cfg(),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_WITH_HANDS_ACTION_SCALE

  # Wire the right-hand grasp site into the reach observation and reward.
  cfg.observations["actor"].terms["ee_to_cube"].params["asset_cfg"].site_names = (
    _GRASP_SITE,
  )
  cfg.rewards["lift"].params["asset_cfg"].site_names = (_GRASP_SITE,)

  for term in (
    "fingertip_friction_slide",
    "fingertip_friction_spin",
    "fingertip_friction_roll",
  ):
    cfg.events[term].params["asset_cfg"].geom_names = _FINGERTIP_GEOMS

  # Place the cube ~1.5-2 m in front and sample a goal ~1-2 m to the side, so the
  # robot must WALK to the cube, then CARRY it to the goal (not just reach).
  command = cfg.commands["lift_height"]
  assert isinstance(command, LiftingCommandCfg)
  command.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=(1.5, 2.0),
    y=(-0.3, 0.3),
    z=(0.02, 0.05),
    yaw=(-3.14, 3.14),
  )
  command.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
    x=(1.5, 2.0),
    y=(-1.5, -1.0),
    z=(0.1, 0.3),
  )
  command.difficulty = "dynamic"

  # End-effector to ground collision sensor targets the right wrist.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = _EE_BODY

  # Add an upright/posture reward so the robot stays standing while it walks and
  # manipulates (locomotion prior). Reuses the velocity task's upright reward.
  from mjlab.tasks.velocity import mdp as vel_mdp

  cfg.rewards["upright"] = RewardTermCfg(
    func=vel_mdp.upright,
    weight=1.0,
    params={
      "std": 0.4,
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso_link",)),
    },
  )

  # More contacts now (feet + fingers + carried object).
  cfg.sim.nconmax = max(cfg.sim.nconmax or 55, 300)

  cfg.viewer.body_name = "torso_link"
  cfg.viewer.distance = 3.0
  cfg.episode_length_s = 30.0

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (10.0, 10.0)

  return cfg
