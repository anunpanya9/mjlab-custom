"""G1 VLA playground env — not an RL task, so it is not registered with the task
registry (which requires an RL runner config). The VLA bridge imports
``g1_vla_env_cfg`` directly."""

from .env_cfgs import ARM_GRIPPER_JOINTS, g1_vla_env_cfg

__all__ = ["ARM_GRIPPER_JOINTS", "g1_vla_env_cfg"]
