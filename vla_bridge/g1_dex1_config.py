from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
  ActionConfig,
  ActionFormat,
  ActionRepresentation,
  ActionType,
  ModalityConfig,
)

# Unitree G1 + Dex1 gripper: dual 7-DOF arms + 2 grippers (16-dim), 3 cameras.
g1_dex1_config = {
  "video": ModalityConfig(
    delta_indices=[0],
    modality_keys=["cam_high", "cam_left_wrist", "cam_right_wrist"],
  ),
  "state": ModalityConfig(
    delta_indices=[0],
    modality_keys=["left_arm", "right_arm", "left_gripper", "right_gripper"],
  ),
  "action": ModalityConfig(
    delta_indices=list(range(0, 16)),
    modality_keys=["left_arm", "right_arm", "left_gripper", "right_gripper"],
    action_configs=[
      ActionConfig(
        rep=ActionRepresentation.RELATIVE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
      ),
      ActionConfig(
        rep=ActionRepresentation.RELATIVE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
      ),
      ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
      ),
      ActionConfig(
        rep=ActionRepresentation.ABSOLUTE,
        type=ActionType.NON_EEF,
        format=ActionFormat.DEFAULT,
      ),
    ],
  ),
  "language": ModalityConfig(
    delta_indices=[0],
    modality_keys=["annotation.human.task_description"],
  ),
}

register_modality_config(g1_dex1_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
