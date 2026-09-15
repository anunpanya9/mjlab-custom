"""mjlab-side client that drives the g1_vla sim with a remote GR00T VLA policy.

Architecture (both processes on the Spark, separate venvs, talking over
localhost so each keeps its own torch build):

    mjlab venv  ── this client ──ZMQ:5555──►  gr00t venv ── run_gr00t_server.py
    (torch cu128)                              (torch cu130, loads checkpoint)

The GR00T policy cannot share mjlab's venv (different torch/CUDA), so instead of
importing gr00t we speak its ZMQ + msgpack wire protocol directly — the client
only needs ``pyzmq`` and ``msgpack``/``msgpack_numpy``.

Start the server first (in the gr00t venv), then run this (in the mjlab venv):

    # terminal 1 — policy server (gr00t venv)
    cd ~/workspace/anun/Isaac-GR00T && source scripts/activate_spark.sh
    PYTORCH_JIT=0 PYTHONPATH=examples/G1Dex1 uv run --no-sync python \
      gr00t/eval/run_gr00t_server.py --model-path ~/workspace/anun/vla_ckpt_g1placebox/checkpoint-2000 \
      --embodiment-tag new_embodiment --modality-config-path examples/G1Dex1/g1_dex1_config.py --port 5555

    # terminal 2 — mjlab client (mjlab venv)
    cd ~/workspace/anun/mjlab-custom && uv run --no-sync python \
      vla_bridge/vla_mjlab_client.py --instruction "Place the red wooden block into the yellow box." \
      --steps 300 --video-path ~/workspace/anun/vla_rollout.mp4

Two modes, one policy: pass a task string with --instruction (language mode); keep
it fixed and it runs autonomously in closed loop.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

# Dataset 16-dim state/action order (meta/info.json names).
DATASET_JOINT_ORDER = [
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
]
STATE_SLICES = {
  "left_arm": slice(0, 7),
  "right_arm": slice(7, 14),
  "left_gripper": slice(14, 15),
  "right_gripper": slice(15, 16),
}


class RemotePolicy:
  """Minimal ZMQ client for GR00T's PolicyServer (get_action endpoint)."""

  def __init__(
    self, host: str = "localhost", port: int = 5555, timeout_ms: int = 60000
  ):
    import msgpack
    import msgpack_numpy as mnp
    import zmq

    self._msgpack = msgpack
    self._mnp = mnp
    self.ctx = zmq.Context()
    self.sock = self.ctx.socket(zmq.REQ)
    self.sock.setsockopt(zmq.RCVTIMEO, timeout_ms)
    self.sock.setsockopt(zmq.SNDTIMEO, timeout_ms)
    self.sock.connect(f"tcp://{host}:{port}")

  def _call(self, endpoint: str, data: dict | None = None):
    req: dict = {"endpoint": endpoint}
    if data is not None:
      req["data"] = data
    packed = self._msgpack.packb(req, default=self._mnp.encode)
    self.sock.send(packed)
    reply = self.sock.recv()
    return self._msgpack.unpackb(reply, object_hook=self._mnp.decode, raw=False)

  def ping(self):
    return self._call("ping")

  def get_action(self, observation: dict):
    return self._call("get_action", {"observation": observation, "options": {}})


def build_camera_obs(env) -> dict:
  """Read the three cameras as dataset-keyed HxWx3 uint8 arrays."""
  out = {}
  for cam, key in (
    ("cam_left_high", "cam_high"),
    ("cam_left_wrist", "cam_left_wrist"),
    ("cam_right_wrist", "cam_right_wrist"),
  ):
    rgb = env.scene[cam].data.rgb  # (B, H, W, 3) uint8
    out[key] = rgb[0].detach().cpu().numpy().astype(np.uint8)
  return out


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument(
    "--instruction", default="Place the red wooden block into the yellow box."
  )
  ap.add_argument("--steps", type=int, default=300)
  ap.add_argument("--video-path", default="vla_rollout.mp4")
  ap.add_argument("--host", default="localhost")
  ap.add_argument("--port", type=int, default=5555)
  ap.add_argument("--device", default="cuda:0")
  args = ap.parse_args()

  os.environ.setdefault("MUJOCO_GL", "egl")
  import imageio

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.manipulation.config.g1_vla.env_cfgs import g1_vla_env_cfg

  cfg = g1_vla_env_cfg()
  cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")
  robot = env.scene["robot"]
  sim_joint_names = list(robot.joint_names)
  action_targets = list(env.action_manager.get_term("joint_pos")._target_names)
  ds_to_action = [action_targets.index(j) for j in DATASET_JOINT_ORDER]
  ds_to_state = [sim_joint_names.index(j) for j in DATASET_JOINT_ORDER]

  policy = RemotePolicy(host=args.host, port=args.port)
  print("ping ->", policy.ping())
  print(f"instruction: {args.instruction!r}")

  env.reset()
  frames = []
  for step in range(args.steps):
    joint_pos = robot.data.joint_pos[0].detach().cpu().numpy()
    state16 = np.array([joint_pos[i] for i in ds_to_state], dtype=np.float32)
    obs = {}
    for k, sl in STATE_SLICES.items():
      obs[f"state.{k}"] = state16[sl][None, :]
    for key, img in build_camera_obs(env).items():
      obs[f"video.{key}"] = img[None, ...]
    obs["annotation.human.task_description"] = args.instruction

    chunk = policy.get_action(obs)
    # chunk[key] shape (batch, horizon, dim); take first horizon step
    ds_action = np.concatenate(
      [
        np.atleast_1d(np.asarray(chunk[f"action.{k}"])[0][0]).reshape(-1)
        if f"action.{k}" in chunk
        else np.atleast_1d(np.asarray(chunk[k])[0][0]).reshape(-1)
        for k in ("left_arm", "right_arm", "left_gripper", "right_gripper")
      ]
    )

    env_action = np.empty(16, dtype=np.float32)
    for ds_i, act_i in enumerate(ds_to_action):
      env_action[act_i] = ds_action[ds_i]
    env.step(torch.as_tensor(env_action, device=args.device).unsqueeze(0))
    frames.append(env.render())
    if step % 25 == 0:
      print(f"  step {step}/{args.steps}")

  imageio.mimsave(args.video_path, frames, fps=30)
  print(f"saved -> {args.video_path}  ({len(frames)} frames)")


if __name__ == "__main__":
  main()
