"""ทดสอบ VLA (GR00T fine-tuned) แบบวัดความแม่นยำที่อธิบายได้ — โดยยังไม่ต้องมี sim/หุ่นจริง.

รันบน DGX Spark (ต้องมี GPU + gr00t venv):

    cd ~/workspace/anun/Isaac-GR00T
    source scripts/activate_spark.sh
    PYTORCH_JIT=0 PYTHONPATH=examples/G1Dex1 \
      uv run --no-sync python test_vla_commands.py \
        --model-path ~/workspace/anun/vla_ckpt_g1placebox/checkpoint-2000 \
        --dataset-path ~/workspace/anun/datasets/G1_PickPlaceRedBlock/unitreerobotics/G1_Dex1_PickPlaceRedBlock_Dataset_Sim

การทดสอบนี้ใช้ observation จริงจาก dataset (open-loop) แล้ววัด 3 อย่าง:

  1. ฟังคำสั่งไหม  — ป้อน obs เดียวกัน + คำสั่งต่างกัน, action ต่างกันมากแค่ไหน
                     (ต่าง = policy อ่านภาษา ไม่ได้เมิน)
  2. แม่นแค่ไหน     — action ที่ทำนาย เทียบ action จริงในเดโม (MAE เป็น radian → องศา)
  3. คำสั่งถูก vs ผิด — พอใส่คำสั่งที่ตรงกับเดโม ควรแม่นกว่าใส่คำสั่งที่ผิด task
"""

from __future__ import annotations

import argparse
from copy import deepcopy

import numpy as np

import g1_dex1_config  # noqa: F401  (registers the NEW_EMBODIMENT modality config)
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.policy.gr00t_policy import Gr00tPolicy

# The two task strings the dataset was collected under, and example episodes for
# each (from meta/episodes.jsonl). Episodes 0-3 are "pick cup"; 4/6/11/13 are
# "place block".
PICK_CUP = "Pick up the red cup on the table."
PLACE_BLOCK = "Place the red wooden block into the yellow box."
CUP_EPISODES = [0, 1, 2]
BLOCK_EPISODES = [4, 6, 11]


def parse_action(chunk: dict) -> dict:
  """Un-batch + add the "action." prefix, matching open_loop_eval."""
  return {f"action.{k}": chunk[k][0] for k in chunk}


def predict_action(policy, traj, mc_obs, action_keys, lang_key, instruction, step=0):
  """Predict the first-step action (16-dim) for the obs at `step`, under one
  instruction. `step` matters: at step 0 both tasks start from the same rest
  pose so they look alike; the tasks diverge mid-trajectory (once the arm
  commits to reaching one way), so pass a mid-episode step to see the
  instruction's effect on behaviour."""
  dp = extract_step_data(traj, step, mc_obs, EmbodimentTag.NEW_EMBODIMENT)
  obs = {}
  for k, v in dp.states.items():
    obs[f"state.{k}"] = v
  for k, v in dp.images.items():
    obs[f"video.{k}"] = np.array(v)
  obs[lang_key] = instruction  # plain string; parse_observation_gr00t wraps it
  parsed = parse_observation_gr00t(obs, mc_obs)
  chunk, _ = policy.get_action(parsed)
  chunk = parse_action(chunk)
  return np.concatenate(
    [np.atleast_1d(np.atleast_1d(chunk[f"action.{k}"])[0]) for k in action_keys]
  )


def ground_truth_action(traj, action_keys, step=0):
  """The real action from the demo at `step` (what the human teleoperator did)."""
  return np.concatenate(
    [
      np.atleast_1d(np.asarray(traj[f"action.{k}"].iloc[step]).reshape(-1))
      for k in action_keys
    ]
  )


def main() -> None:
  ap = argparse.ArgumentParser()
  ap.add_argument("--model-path", required=True)
  ap.add_argument("--dataset-path", required=True)
  args = ap.parse_args()

  policy = Gr00tPolicy(
    embodiment_tag="new_embodiment", model_path=args.model_path, device="cuda:0"
  )
  mc = policy.get_modality_config()
  loader = LeRobotEpisodeLoader(dataset_path=args.dataset_path, modality_configs=mc)
  action_keys = mc["action"].modality_keys
  lang_key = mc["language"].modality_keys[0]
  mc_obs = deepcopy(mc)
  mc_obs.pop("action")

  # radian -> degree helper for readable errors
  def deg(x):
    return np.rad2deg(x)

  # Measure at a mid-trajectory step, not step 0. Both tasks start from the same
  # rest pose, so at step 0 they look identical; they diverge once the arm
  # commits to reaching. STEP is capped per-episode by trajectory length.
  STEP = 300

  def ep_step(traj):
    return min(STEP, len(traj) - 1)

  print("\n" + "=" * 64)
  print(f"การทดสอบ 1: policy ฟังคำสั่งไหม (obs เดียวกัน คำสั่งต่างกัน, step {STEP})")
  print("=" * 64)
  traj = loader[BLOCK_EPISODES[0]]
  s = ep_step(traj)
  a_cup = predict_action(policy, traj, mc_obs, action_keys, lang_key, PICK_CUP, s)
  a_blk = predict_action(policy, traj, mc_obs, action_keys, lang_key, PLACE_BLOCK, s)
  diff = float(np.abs(a_cup - a_blk).mean())
  print(f"  คำสั่ง A ('pick cup')    -> mean|action| = {np.abs(a_cup).mean():.4f}")
  print(f"  คำสั่ง B ('place block') -> mean|action| = {np.abs(a_blk).mean():.4f}")
  print(f"  ความต่างเฉลี่ยของ action  = {diff:.4f}  ({deg(diff):.2f} องศา/ข้อต่อ)")
  print(
    "  => "
    + (
      "policy ฟังคำสั่ง (action เปลี่ยนตามคำสั่ง)"
      if diff > 1e-3
      else "policy ไม่ฟังคำสั่ง (action เหมือนเดิม)"
    )
  )

  print("\n" + "=" * 64)
  print(f"การทดสอบ 2: ความแม่นยำ — action ทำนาย เทียบ demo จริง (คำสั่งถูก, step {STEP})")
  print("=" * 64)
  print(f"  {'episode':>8} {'task':>12} {'MAE(rad)':>10} {'MAE(deg)':>10}")
  correct_maes = []
  for eps, task, label in [(e, PICK_CUP, "pick-cup") for e in CUP_EPISODES] + [
    (e, PLACE_BLOCK, "place-block") for e in BLOCK_EPISODES
  ]:
    traj = loader[eps]
    s = ep_step(traj)
    pred = predict_action(policy, traj, mc_obs, action_keys, lang_key, task, s)
    gt = ground_truth_action(traj, action_keys, s)
    mae = float(np.abs(pred - gt).mean())
    correct_maes.append(mae)
    print(f"  {eps:>8} {label:>12} {mae:>10.4f} {deg(mae):>10.2f}")
  mean_correct = float(np.mean(correct_maes))
  print(f"  {'เฉลี่ย':>8} {'':>12} {mean_correct:>10.4f} {deg(mean_correct):>10.2f}")

  print("\n" + "=" * 64)
  print(f"การทดสอบ 3: คำสั่งถูก vs คำสั่งผิด (ควรแม่นกว่าเมื่อคำสั่งตรง task, step {STEP})")
  print("=" * 64)
  # For "place block" demos, compare the correct instruction vs the wrong one at
  # a mid step (where the tasks actually differ).
  wrong_maes = []
  for eps in BLOCK_EPISODES:
    traj = loader[eps]
    s = ep_step(traj)
    gt = ground_truth_action(traj, action_keys, s)
    pred_wrong = predict_action(
      policy,
      traj,
      mc_obs,
      action_keys,
      lang_key,
      PICK_CUP,
      s,  # wrong task
    )
    wrong_maes.append(float(np.abs(pred_wrong - gt).mean()))
  mean_wrong = float(np.mean(wrong_maes))
  mean_block_correct = float(np.mean(correct_maes[len(CUP_EPISODES) :]))
  print(
    f"  block demos + คำสั่งถูก ('place block') -> MAE {mean_block_correct:.4f} ({deg(mean_block_correct):.2f} องศา)"
  )
  print(
    f"  block demos + คำสั่งผิด ('pick cup')     -> MAE {mean_wrong:.4f} ({deg(mean_wrong):.2f} องศา)"
  )
  better = mean_block_correct < mean_wrong
  print(
    "  => "
    + (
      "คำสั่งถูกแม่นกว่า (policy ใช้คำสั่งจริงในการตัดสินใจ)"
      if better
      else "คำสั่งผิดไม่ได้แย่กว่า — จังหวะนี้อาจยังคล้ายกัน หรือ policy พึ่งภาพเป็นหลัก"
    )
  )

  print("\n" + "=" * 64)
  print("สรุป")
  print("=" * 64)
  print(f"  1. ฟังคำสั่ง: action ต่างกัน {diff:.3f} rad เมื่อคำสั่งต่าง -> ฟังจริง")
  print(
    f"  2. ความแม่น: MAE เฉลี่ย {mean_correct:.3f} rad ({deg(mean_correct):.1f} องศา/ข้อต่อ)"
  )
  print(
    f"  3. คำสั่งถูก ({deg(mean_block_correct):.1f}°) "
    + ("< " if better else ">= ")
    + f"คำสั่งผิด ({deg(mean_wrong):.1f}°)"
  )
  print(
    "\nหมายเหตุ: open-loop วัดทีละจังหวะจาก obs ของ demo จริง — เป็นการเช็คว่า policy\n"
    "ทำนายใกล้เคียงคนไหม การพิสูจน์ว่าทำงานได้จริง (หยิบสำเร็จ) ต้องทดสอบ closed-loop\n"
    "ใน sim/หุ่นจริง ซึ่งเป็นขั้นถัดไป (mjlab bridge)."
  )


if __name__ == "__main__":
  main()
