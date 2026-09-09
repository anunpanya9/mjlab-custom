from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import g1_lift_cube_env_cfg, g1_lift_cube_vision_env_cfg
from .rl_cfg import g1_lift_cube_ppo_runner_cfg, g1_lift_cube_vision_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-G1",
  env_cfg=g1_lift_cube_env_cfg(),
  play_env_cfg=g1_lift_cube_env_cfg(play=True),
  rl_cfg=g1_lift_cube_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-G1-Rgb",
  env_cfg=g1_lift_cube_vision_env_cfg(cam_type="rgb"),
  play_env_cfg=g1_lift_cube_vision_env_cfg(cam_type="rgb", play=True),
  rl_cfg=g1_lift_cube_vision_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-G1-Depth",
  env_cfg=g1_lift_cube_vision_env_cfg(cam_type="depth"),
  play_env_cfg=g1_lift_cube_vision_env_cfg(cam_type="depth", play=True),
  rl_cfg=g1_lift_cube_vision_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
