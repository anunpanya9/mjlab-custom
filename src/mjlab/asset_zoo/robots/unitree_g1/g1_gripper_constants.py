"""G1 with simple parallel-jaw grippers (Dex1-style), for the VLA bridge.

The Unitree ``G1_Dex1_PickPlaceRedBlock`` teleop dataset controls a 16-DOF
action space: the two 7-DOF arms plus one scalar per gripper. mjlab ships G1
with the Dex3-1 three-fingered hands (14 finger DOF), which does not match. To
run that VLA policy in mjlab we need an embodiment whose action space is exactly
those 16 dims.

Rather than model the real Dex1 gripper mesh, this attaches a **simple
parallel-jaw gripper** (a base plus two sliding box fingers) to each wrist. Each
gripper adds one actuated slide DOF, so the full robot exposes 14 arm joints +
2 gripper joints = 16, matching the dataset. This is enough to drive and test
the VLA in sim; it is not a geometric replica of the real hand.
"""

from __future__ import annotations

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  DAMPING_RATIO,
  FULL_COLLISION,
  G1_ACTUATOR_4010,
  G1_ACTUATOR_5020,
  G1_ACTUATOR_7520_14,
  G1_ACTUATOR_7520_22,
  G1_ACTUATOR_ANKLE,
  G1_ACTUATOR_WAIST,
  G1_XML,
  NATURAL_FREQ,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec import get_free_joint

# Parallel-jaw gripper geometry (metres). The two fingers slide along the local
# y-axis of the wrist; travel 0 = closed, +GRIPPER_OPEN = fully open.
_GRIPPER_FINGER = (0.008, 0.006, 0.03)  # half-sizes (x deep, y thick, z long)
_GRIPPER_OFFSET_Z = 0.05  # how far past the wrist the fingers start
_GRIPPER_OPEN = 0.04  # max half-opening of each finger


def _add_gripper(spec: mujoco.MjSpec, wrist_body_name: str, side: str) -> None:
  """Attach a two-finger parallel-jaw gripper to one wrist body.

  Each gripper exposes exactly **one** actuated DOF (``{side}_gripper_joint``, the
  driving finger). The opposite finger mirrors it through a MuJoCo joint equality
  constraint, so the pair opens and closes symmetrically but the action space
  gains only one dimension per hand — matching the dataset's single gripper
  scalar. Positive joint value = open.
  """
  wrist = spec.body(wrist_body_name)
  hx, hy, hz = _GRIPPER_FINGER

  # Driving finger (actuated).
  drive = wrist.add_body(
    name=f"{side}_gripper_drive_finger", pos=(0.0, 0.02, _GRIPPER_OFFSET_Z)
  )
  drive.add_joint(
    name=f"{side}_gripper_joint",
    type=mujoco.mjtJoint.mjJNT_SLIDE,
    axis=(0.0, 1.0, 0.0),
    range=(0.0, _GRIPPER_OPEN),
  )
  drive.add_geom(
    name=f"{side}_gripper_drive_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(hx, hy, hz),
    rgba=(0.2, 0.2, 0.22, 1.0),
  )

  # Mirror finger (follows the driving finger, opposite direction).
  mirror = wrist.add_body(
    name=f"{side}_gripper_mirror_finger", pos=(0.0, -0.02, _GRIPPER_OFFSET_Z)
  )
  mirror.add_joint(
    name=f"{side}_gripper_mirror_joint",
    type=mujoco.mjtJoint.mjJNT_SLIDE,
    axis=(0.0, -1.0, 0.0),
    range=(0.0, _GRIPPER_OPEN),
  )
  mirror.add_geom(
    name=f"{side}_gripper_mirror_geom",
    type=mujoco.mjtGeom.mjGEOM_BOX,
    size=(hx, hy, hz),
    rgba=(0.2, 0.2, 0.22, 1.0),
  )

  # Couple mirror = drive (both slide outward together as the value grows).
  eq = spec.add_equality()
  eq.type = mujoco.mjtEq.mjEQ_JOINT
  eq.name1 = f"{side}_gripper_mirror_joint"
  eq.name2 = f"{side}_gripper_joint"
  eq.objtype = mujoco.mjtObj.mjOBJ_JOINT
  # polycoef: mirror = c0 + c1*drive (+ higher order). c1=1 => 1:1 follow.
  eq.data[:5] = [0.0, 1.0, 0.0, 0.0, 0.0]


def get_spec_with_grippers() -> mujoco.MjSpec:
  """Base G1 (no hands) with a parallel-jaw gripper on each wrist."""
  spec = mujoco.MjSpec.from_file(str(G1_XML))
  _add_gripper(spec, "left_wrist_yaw_link", "left")
  _add_gripper(spec, "right_wrist_yaw_link", "right")
  return spec


def get_spec_with_grippers_fixed_base() -> mujoco.MjSpec:
  """Gripper G1 with the pelvis welded to the world (stationary manipulation)."""
  spec = get_spec_with_grippers()
  free_joint = get_free_joint(spec)
  if free_joint is not None:
    spec.delete(free_joint)
  return spec


# Gripper actuator: same critically-overdamped PD target as the arm/hands, tuned
# for the light sliding fingers.
GRIPPER_ARMATURE = 1.0e-5
GRIPPER_STIFFNESS = GRIPPER_ARMATURE * NATURAL_FREQ**2
GRIPPER_DAMPING = 2.0 * DAMPING_RATIO * GRIPPER_ARMATURE * NATURAL_FREQ

G1_ACTUATOR_GRIPPER = BuiltinPositionActuatorCfg(
  # Only the driving joint of each hand is actuated; the mirror joint follows via
  # the equality constraint. This keeps the gripper action space at 2 (one per
  # hand), matching the dataset. Matches "left_gripper_joint"/"right_gripper_joint"
  # but NOT "*_gripper_mirror_joint".
  target_names_expr=("left_gripper_joint", "right_gripper_joint"),
  stiffness=GRIPPER_STIFFNESS,
  damping=GRIPPER_DAMPING,
  effort_limit=20.0,
  armature=GRIPPER_ARMATURE,
)

G1_WITH_GRIPPERS_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    G1_ACTUATOR_5020,
    G1_ACTUATOR_7520_14,
    G1_ACTUATOR_7520_22,
    G1_ACTUATOR_4010,
    G1_ACTUATOR_WAIST,
    G1_ACTUATOR_ANKLE,
    G1_ACTUATOR_GRIPPER,
  ),
  soft_joint_pos_limit_factor=0.9,
)

# Stationary manipulation pose: arms reaching forward, ready to grasp (same as
# the Dex3 manipulation keyframe; grippers rest closed at 0).
GRIPPER_MANIPULATION_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0),
  joint_pos={
    ".*_shoulder_pitch_joint": -0.8,
    "left_shoulder_roll_joint": 0.1,
    "right_shoulder_roll_joint": -0.1,
    ".*_elbow_joint": 1.2,
  },
  joint_vel={".*": 0.0},
)


def get_g1_with_grippers_fixed_base_robot_cfg() -> EntityCfg:
  """G1 with parallel-jaw grippers, pelvis welded — the embodiment whose 16-DOF
  action space (14 arm + 2 gripper) matches the Dex1 teleop dataset, for running
  the fine-tuned VLA policy in mjlab."""
  return EntityCfg(
    init_state=GRIPPER_MANIPULATION_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec_with_grippers_fixed_base,
    articulation=G1_WITH_GRIPPERS_ARTICULATION,
  )
