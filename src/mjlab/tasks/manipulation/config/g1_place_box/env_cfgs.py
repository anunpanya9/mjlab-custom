"""Stationary G1 pick-and-place task: put a cube into a box.

The G1 is welded at the pelvis (fixed-base) so only the arms and Dex3-1 hands
move, exactly like the lift-cube task. A red cube rests on a table; the task is
to grasp it with the right hand and release it into an open-top yellow box that
sits on the far side of the table. This mirrors the Unitree
``G1_Dex1_PickPlaceRedBlock`` teleop dataset ("Place the red wooden block into
the yellow box"), but is trained with RL in our own sim rather than imitation,
so the Dex3 hand and 43-DOF action space are used as-is.

It reuses the lift-cube stack: the ``lift_height`` command's target is simply
placed inside the box (a low z near the box floor) instead of hovering in the
air, and a physical box entity is added at that target so the cube must actually
end up resting inside it.
"""

import mujoco

from mjlab.asset_zoo.robots import (
  G1_WITH_HANDS_ACTION_SCALE,
  get_g1_with_hands_fixed_base_robot_cfg,
)
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.manipulation.config.g1.env_cfgs import (
  _EE_BODY,
  _FINGERTIP_GEOMS,
  _GRASP_SITE,
  _TABLE_HEIGHT,
  get_cube_spec,
  get_table_spec,
)
from mjlab.tasks.manipulation.lift_cube_env_cfg import make_lift_cube_env_cfg
from mjlab.tasks.manipulation.mdp import LiftingCommandCfg

# The box sits on the table, to the robot's left-front (matching the dataset's
# yellow target box). Its opening is centered here in the robot base frame.
_BOX_CENTER_X = 0.45
_BOX_CENTER_Y = 0.25
# Interior half-extent of the box opening and wall/floor thickness.
_BOX_HALF = 0.06
_BOX_WALL = 0.008
_BOX_DEPTH = 0.05  # interior depth (wall height above the floor)


def get_box_spec(
  half: float = _BOX_HALF,
  wall: float = _BOX_WALL,
  depth: float = _BOX_DEPTH,
  rgba: tuple[float, float, float, float] = (0.95, 0.85, 0.1, 1.0),
) -> mujoco.MjSpec:
  """An open-top box: a thin floor plus four thin walls, welded to the world.

  The box is anchored on the table so the cube can be dropped into it. The
  interior clear span is ``2*half`` on each side and ``depth`` tall.
  """
  spec = mujoco.MjSpec()
  floor_z = _TABLE_HEIGHT + wall
  body = spec.worldbody.add_body(
    name="box", pos=(_BOX_CENTER_X, _BOX_CENTER_Y, floor_z)
  )
  # Floor.
  body.add_geom(
    name="box_floor",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(half + wall, half + wall, wall),
    rgba=rgba,
  )
  wall_z = depth / 2.0 + wall
  wall_specs = (
    ("box_wall_px", (wall, half + wall, depth / 2.0), (half + wall, 0.0, wall_z)),
    ("box_wall_nx", (wall, half + wall, depth / 2.0), (-half - wall, 0.0, wall_z)),
    ("box_wall_py", (half + wall, wall, depth / 2.0), (0.0, half + wall, wall_z)),
    ("box_wall_ny", (half + wall, wall, depth / 2.0), (0.0, -half - wall, wall_z)),
  )
  for name, size, pos in wall_specs:
    body.add_geom(
      name=name,
      type=mujoco.mjtGeom.mjGEOM_BOX,
      size=size,
      pos=pos,
      rgba=rgba,
    )
  return spec


def g1_place_box_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = make_lift_cube_env_cfg()

  cfg.scene.entities = {
    "robot": get_g1_with_hands_fixed_base_robot_cfg(),
    "table": EntityCfg(spec_fn=get_table_spec),
    "box": EntityCfg(spec_fn=get_box_spec),
    "cube": EntityCfg(spec_fn=get_cube_spec),
  }

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_WITH_HANDS_ACTION_SCALE

  # Wire the G1 right-hand grasp site into the reach observation and reward.
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

  # Cube starts on the table to the robot's right (opposite side from the box),
  # so the policy must carry it across to the box rather than nudge it.
  cube_z = _TABLE_HEIGHT + 0.02
  command = cfg.commands["lift_height"]
  assert isinstance(command, LiftingCommandCfg)
  command.object_pose_range = LiftingCommandCfg.ObjectPoseRangeCfg(
    x=(0.35, 0.55),
    y=(-0.3, -0.05),
    z=(cube_z, cube_z + 0.02),
    yaw=(-3.14, 3.14),
  )
  # The goal is inside the box: a fixed point just above the box floor. Using a
  # tight range keeps the target stable so success means "cube resting in box".
  goal_z = _TABLE_HEIGHT + 2 * _BOX_WALL + 0.02
  command.difficulty = "dynamic"
  command.target_position_range = LiftingCommandCfg.TargetPositionRangeCfg(
    x=(_BOX_CENTER_X - 0.02, _BOX_CENTER_X + 0.02),
    y=(_BOX_CENTER_Y - 0.02, _BOX_CENTER_Y + 0.02),
    z=(goal_z, goal_z + 0.01),
  )
  # A cube (0.02 half-extent) inside a box needs a slightly looser success
  # threshold than the bare lift task's 0.05.
  command.success_threshold = 0.06

  # End-effector to ground/table collision sensor targets the right wrist.
  assert cfg.scene.sensors is not None
  for sensor in cfg.scene.sensors:
    if sensor.name == "ee_ground_collision":
      assert isinstance(sensor, ContactSensorCfg)
      sensor.primary.pattern = _EE_BODY

  # More contacts now that fingers, a table, and a box are in the scene.
  cfg.sim.nconmax = max(cfg.sim.nconmax or 55, 300)

  cfg.viewer.body_name = "torso_link"

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.curriculum = {}
    assert cfg.commands is not None
    cfg.commands["lift_height"].resampling_time_range = (6.0, 6.0)

  return cfg
