from mjlab.tasks.manipulation.rl import ManipulationOnPolicyRunner
from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import g1_locomanip_env_cfg
from .rl_cfg import g1_locomanip_ppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-LocoManip-Unitree-G1",
  env_cfg=g1_locomanip_env_cfg(),
  play_env_cfg=g1_locomanip_env_cfg(play=True),
  rl_cfg=g1_locomanip_ppo_runner_cfg(),
  runner_cls=ManipulationOnPolicyRunner,
)
