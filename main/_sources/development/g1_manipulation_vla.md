# G1 Mobile Manipulation → VLA — Development Guide

Internal working document for the project that takes mjlab's manipulation
stack from a fixed-arm lift task all the way to a Vision–Language–Action (VLA)
policy for a Unitree G1 doing mobile manipulation ("walk over, pick it up, hand
it over").

This is a living document. Update it as each phase lands, so a future session
(or a new contributor) can pick up exactly where the last one left off.

---

## 1. Goal

A single policy that takes an RGB camera view plus a natural-language
instruction — e.g. *"walk over and hand me the red cube"* — and outputs
whole-body actions for a G1 that walks, grasps with a dexterous hand, and
hands the object over.

- **Robot:** Unitree G1 (29 DOF) + two Unitree Dex3-1 hands (14 DOF, 43 total).
- **Task family:** loco-manipulation (walk → grasp → carry → hand over).
- **End state:** full VLA, language-conditioned, trained by imitation.
- **Scope:** full research project, single researcher.

## 2. Guiding strategy

Language-conditioned RL from reward alone does not scale to open instructions.
The field (OpenVLA, π0, GR00T, RoboCat) converges on **imitation**. So the
spine of the whole project is:

```
RL experts  →  roll out demonstrations (image + instruction + action)  →  distill into the VLA
```

mjlab's job for most of the project is the **data engine** and the **eval
harness**. The VLA model itself is trained offline (a separate training stack).

## 3. Phase roadmap

The agreed order is sequential — legs go on **before** the VLA. (An
alternative "VLA-first on a fixed arm" order was considered and rejected; we
keep the sequential plan.)

| Phase | Goal | Thread | Status |
|-------|------|--------|--------|
| 0 | Reproduce baselines (YAM lift-cube, G1 velocity) | action | see §4 |
| 1 | Dex3 hands on G1 + stationary grasp task | action | **hands + task built; training next** |
| 2 | Vision-based grasping (camera replaces privileged state) | vision | **env built; training next** |
| 3 | Legs on — walk, grasp, carry, hand over | action | todo |
| 4 | Language + distillation into the VLA | language | todo |

### Known hard parts (design around these)
1. **Whole-body loco-manipulation (P3)** — walking while manipulating is the
   crux; reward shaping must not trade one off against the other. Lean on the
   `tracking` (motion-imitation) task as a whole-body prior.
2. **Render throughput (P2+)** — on-GPU rendering across many envs caps
   parallelism. Prototype at low resolution; profile before committing P3.
3. **Contact-rich grasping (P1)** — the dexterous hand adds DOF and contacts
   that stress the solver (`nconmax`/`njmax`).
4. **Instruction generalization (P4)** — vary phrasing, add distractors, hold
   out unseen instruction/layout combos in eval.

## 4. Reusable building blocks already in mjlab

- **Camera sensor** with `rgb`/`depth`/`segmentation`
  (`src/mjlab/sensor/camera_sensor.py`) and the CNN + spatial-softmax vision
  policy path in the RL runner (`src/mjlab/rl/spatial_softmax.py`).
- **YAM lift-cube task** — the manipulation template we clone. Staged reward
  (reaching → bringing → precise), fingertip friction domain randomization,
  contact sensors, curriculum. See
  `src/mjlab/tasks/manipulation/lift_cube_env_cfg.py` and
  `src/mjlab/tasks/manipulation/config/yam/`.
- **`MultiCubeLiftingCommand`** — samples a random target object among several;
  the seed of language grounding (which object the instruction names).
- **`tracking` task** — motion imitation, the whole-body prior for P3.
- **Manager-based env** — `observations / actions / rewards / commands /
  events / terminations / curriculum`, all composable.

## 5. Key architecture notes (learned while building)

- **Fixed vs floating base is decided by the MJCF**, not a config flag. An
  entity with a freejoint at its root is floating-base; without one it is
  fixed-base. mjlab auto-wraps fixed-base entities in a mocap body
  (`auto_wrap_fixed_base_mocap` in `src/mjlab/utils/spec.py`) so they can be
  positioned per-env — but positioning only happens if a reset event runs
  (e.g. `reset_root_state_uniform`).
- **YAM is fixed-base (8 joints); G1 is floating-base (43 joints).** For the
  Phase-1 stationary grasp we pin the pelvis by building a G1 spec variant with
  the pelvis freejoint removed (see §7). This isolates manipulation from
  locomotion, exactly as planned.
- **Actuators/keyframes live in Python**, not the MJCF. mjlab strips the
  `<actuator>`/`<keyframe>` blocks from the Menagerie XMLs and defines them via
  `BuiltinPositionActuatorCfg` + `EntityCfg.InitialStateCfg` in
  `g1_constants.py`. Any new MJCF must follow this (keep `<sensor>` and
  `<contact>`, drop `<actuator>`/`<keyframe>`).
- **Collision geoms must be named `*_collision`** so mjlab's `CollisionCfg`
  (which matches `.*_collision`) applies contype/condim to them. Menagerie
  leaves hand collision geoms unnamed; the build script names them.
- **Tasks are registered** via `register_mjlab_task(task_id, env_cfg,
  play_env_cfg, rl_cfg, runner_cls)` in each config package's `__init__.py`.
  The package must be imported for the task to appear (see
  `src/mjlab/tasks/manipulation/__init__.py`).

## 6. Phase 1 — progress log

### 6.1 Dex3 hands on G1 — DONE

Added a G1 variant with two Unitree Dex3-1 three-fingered hands (7 DOF each,
14 total → 43 DOF).

**Files:**
- `src/mjlab/asset_zoo/robots/unitree_g1/xmls/g1_with_hands.xml` — derived from
  mjlab's `g1.xml` (not copied from Menagerie) so the leg/arm bodies, sensors
  and contact excludes stay byte-for-byte identical. Only the two
  `wrist_yaw_link` subtrees change: the rubber hand is replaced by the
  Menagerie palm + three-finger subtree (with Menagerie's corrected wrist_yaw
  inertial, which folds in palm mass), and a `*_grasp_site` is added at each
  palm for manipulation observations/rewards.
- 16 hand mesh STLs in `.../xmls/assets/`.
- `g1_constants.py` — `get_g1_with_hands_robot_cfg()`,
  `G1_WITH_HANDS_ARTICULATION`, `G1_WITH_HANDS_ACTION_SCALE`, and the Dex3
  finger actuators (`G1_ACTUATOR_HAND_THUMB_0`, `G1_ACTUATOR_HAND_FINGERS`).
- Exported from `src/mjlab/asset_zoo/robots/__init__.py`.
- Tests in `tests/test_g1_constants.py` and `tests/test_asset_zoo.py`.

**Decisions to revisit:**
- Dex3 rotor specs are not published by Unitree. We approximated reflected
  inertia (`HAND_ARMATURE = 1e-5`) and tuned stiffness/damping to the same
  critically-overdamped 10 Hz target as the arm. Effort limits (2.45 / 1.4 Nm)
  come straight from the MJCF `actuatorfrcrange`. **These may need retuning if
  the grasp is unstable during training.**
- The grasp site sits ~0.09 m in front of the wrist, between thumb and index
  (verified ~0.07 m from the thumb tip). Adjust if grasps consistently miss.

**How the XML was built:** `scratchpad/build_g1_with_hands.py` (parses both
MJCFs with ElementTree, grafts the hand subtrees, names collision geoms, adds
grasp sites, validates by compiling). Re-run it if the base `g1.xml` changes.

### 6.2 Stationary G1 grasp task — BUILT (training pending)

Registered task **`Mjlab-Lift-Cube-G1`**. The G1 is pinned at the pelvis and
reaches for a cube on a table with its right hand.

**Files:**
- `src/mjlab/asset_zoo/robots/unitree_g1/g1_constants.py` —
  `get_spec_with_hands_fixed_base()` (removes the pelvis freejoint),
  `get_g1_with_hands_fixed_base_robot_cfg()`, and `MANIPULATION_KEYFRAME`
  (arms-forward ready pose).
- `src/mjlab/tasks/manipulation/config/g1/{env_cfgs,rl_cfg,__init__}.py` —
  cloned from the YAM config. Task auto-discovers via `mjlab.tasks` package
  walking (no central import needed).

**Wiring decisions:**
- Single right hand for Phase 1 (`right_grasp_site`, right fingertip geoms).
  Bimanual comes later.
- Added a static `table` entity (box top at 0.8 m). The cube spawns on the
  table within reach: `object_pose_range` x∈(0.35,0.55), y∈(-0.3,0.0),
  z≈table+cube. **Reasoning:** a standing G1's hands reach ~0.85 m height in
  front; picking off the floor would need deep bending, which is out of scope
  for a pinned base.
- `MANIPULATION_KEYFRAME` uses negative shoulder pitch (arms swing forward) and
  elbow flex 1.2; `pos=(0,0,0)` because the pelvis body already carries a
  +0.793 m z-offset in the MJCF. Verified in sim: grasp site lands ~(0.40,
  0.85) in front of the robot, right above the cube (see render).
- `nconmax` raised to 200 (fingers + table add contacts).

**Verified:** env builds, resets, and steps with zero actions on CPU (obs dim
135, finite rewards, no NaN). Rendered scene confirms the geometry (robot
reaching over the table, cube in range). YAM tasks still register/load (no
regression). `make check` + tests pass.

**Not yet done:** actual training (needs a GPU). Sanity-check first with
`uv run play Mjlab-Lift-Cube-G1 --agent zero` / `--agent random`, then train.
The open reward question (does the hand need an explicit finger-close/hold
term?) can only be answered once training runs — log the result here.

## 6.3 Phase 2 — vision-based grasping — BUILT (training pending)

Registered **`Mjlab-Lift-Cube-G1-Rgb`** and **`Mjlab-Lift-Cube-G1-Depth`**.
The actor now sees the object through a torso head camera instead of the
privileged cube pose.

**Files:** `config/g1/env_cfgs.py` (`g1_lift_cube_vision_env_cfg`) and
`config/g1/rl_cfg.py` (`g1_lift_cube_vision_ppo_runner_cfg`, with the
spatial-softmax CNN encoder).

**Design:**
- **Head camera** on `robot/torso_link` at pos (0.1, 0, 0.12), a fixed
  look-down quaternion (-0.4922, -0.332, 0.4499, 0.6671), fovy 75, 64×64. The
  quat was found by MuJoCo `targetbody` targeting of the table center, then
  read back for a static `CameraSensorCfg` (which only takes pos/quat).
  `scratchpad/test_head_cam3.py` reproduces it. Renders show the cube and both
  hands in frame. **Phase 3 must revisit this** once the torso moves — a fixed
  quat only works with a pinned base.
- **Asymmetric actor-critic:** the actor drops the privileged `ee_to_cube` /
  `cube_to_goal` terms and gains the commanded `goal_position` (in hand frame)
  plus the camera group; the critic keeps the full privileged state plus the
  camera. Verified obs dims: actor 132, critic 135, camera (B, 3, 64, 64) for
  RGB / (B, 1, 64, 64) for depth.
- **RGB cube-color randomization** (`dr.geom_rgba` on reset) so the policy
  can't memorize a fixed color.

**Verified:** both RGB and depth envs build, reset, and step on CPU; on-GPU
rendering runs (warp render kernel compiles); camera obs is in [0,1] with no
NaN; the saved 64×64 view frames the workspace correctly. `make check` passes.

**Open questions for training:** 64×64 may be too coarse to localize a 2 cm
cube — bump resolution if the policy can't find it (watch throughput). The
asymmetric setup should help PPO bootstrap from the privileged critic.

## 7. Phase 1 — stationary grasp task plan

Clone the YAM lift-cube task for the G1-with-hands, pelvis pinned.

1. **Fixed-base G1 spec** — add `get_spec_with_hands_fixed_base()` (or similar)
   to `g1_constants.py` that removes the pelvis freejoint so mjlab treats the
   robot as fixed-base and mocap-wraps it. Provide a matching
   `get_g1_with_hands_fixed_base_robot_cfg()`.
2. **Task config package** `src/mjlab/tasks/manipulation/config/g1/`:
   - `env_cfgs.py` — `g1_lift_cube_env_cfg(play=False)`, cloning
     `make_lift_cube_env_cfg()`. Wire up: robot = G1 fixed-base with hands;
     `grasp_site` for the `ee_to_cube` observation and `lift` reward
     (use one hand to start — e.g. `right_grasp_site`); fingertip friction
     geoms = the Dex3 fingertip collision geoms; action scale =
     `G1_WITH_HANDS_ACTION_SCALE`; collision sensor pattern for
     end-effector→ground; viewer body = `torso_link` (or a hand body).
   - `rl_cfg.py` — clone `yam_lift_cube_ppo_runner_cfg()`.
   - `__init__.py` — `register_mjlab_task("Mjlab-Lift-Cube-G1", ...)`.
   - Register the package import in the manipulation task `__init__.py`.
3. **Verify** with dummy agents before training:
   `uv run play Mjlab-Lift-Cube-G1 --agent zero` and `--agent random`.

**Open design question for grasp reward:** the YAM reward rewards bringing the
EE site to the cube and lifting. A dexterous hand also needs a *closing* signal
(finger contact + hold). Start by reusing the YAM staged reward as-is (site
reaches cube, cube lifts) and only add an explicit finger-close/hold reward if
the hand won't grip. Log the outcome here.

## 8. Commands cheat-sheet

```sh
# List tasks
uv run list-envs                       # or: uv run python -m ... (see scripts)

# Sanity-check MDP with dummy agents (no training).
# On macOS the default/native viewer needs `mjpython` (it uses
# mujoco.viewer.launch_passive, which must own the main thread). Pass
# `--viewer viser` to get a browser-based viewer under plain `uv run` instead:
uv run play Mjlab-Lift-Cube-G1 --agent zero --viewer viser --num-envs 1
uv run play Mjlab-Lift-Cube-G1 --agent random --viewer viser --num-envs 1
# (On Linux with a display, `--agent zero` alone uses the native viewer.)

# Train (needs NVIDIA GPU)
uv run train Mjlab-Lift-Cube-G1 --env.scene.num-envs 4096

# Checks before committing
make check      # format + lint + type
make test-fast  # tests minus slow ones
```

**macOS note:** `RuntimeError: launch_passive requires ... mjpython` means the
native viewer was selected. It is not a task bug — use `--viewer viser`, or run
under `mjpython` if you specifically want the native GUI. Training and headless
eval are unaffected.

## 9. Changelog of this document

- Phase 1.1 (hands on G1) complete. Phase 1.2 (stationary grasp task)
  built and verified (env builds/steps, viser viewer runs on macOS); training
  pending a GPU.
