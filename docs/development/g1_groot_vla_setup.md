# G1 VLA with NVIDIA Isaac GR00T on DGX Spark — setup log

Living document tracking the effort to fine-tune a Vision-Language-Action (VLA)
model for the Unitree G1 pick-and-place task on the DGX Spark (GB10 Blackwell),
using NVIDIA Isaac GR00T N1.7 and the Unitree teleop dataset.

## Deploying the VLA back into mjlab sim (planned)

Goal: run the fine-tuned VLA closed-loop in mjlab so we can try it (and issue
language commands) without the real robot. Two hard facts shape the plan:

1. **The Mac can't run GR00T** (no CUDA / flash-attn). So inference stays on the
   Spark as a **policy server** (`gr00t/policy/server_client.py`, `PolicyServer` /
   `PolicyClient`), and the mjlab sim is the **client** — it sends observations
   (3 camera images + 16-dim state + a language string) over the network and gets
   back an action chunk. The `PolicyClient` speaks to a `PolicyServer` on host/port.

2. **mjlab has no G1 + Dex1 gripper asset.** It ships G1 base and G1+Dex3 (three
   fingers, 43 DOF) only. The dataset's 16-dim state/action is:
   `[0-6] left arm` (shoulder pitch/roll/yaw, elbow, wrist roll/pitch/yaw),
   `[7-13] right arm` (same), `[14] left gripper`, `[15] right gripper`.
   The **14 arm joints match mjlab's G1 arm joints exactly**
   (`{left,right}_{shoulder_pitch,shoulder_roll,shoulder_yaw,elbow,wrist_roll,wrist_pitch,wrist_yaw}_joint`).
   The gap is only the **2 grippers** — need to add a 2-finger gripper (or map the
   gripper scalar to a simple parallel-jaw joint) at each wrist.

Plan for the mjlab Dex1 env (to build):
- Fixed-base G1 with the 14 arm joints (reuse existing G1 asset) + a 2-finger
  gripper per wrist so the action space is exactly the 16 dims the VLA outputs.
- 3 cameras matching the dataset: a head/front camera (`cam_left_high`) and two
  wrist cameras (`cam_left_wrist`, `cam_right_wrist`), rendered at the VLA's
  expected resolution.
- A client loop: render cameras + read joint state → build obs dict → send to the
  Spark policy server with a language instruction → apply returned action chunk →
  step sim → repeat. This gives both modes: **language mode** (change the
  instruction string) and **autonomous mode** (fixed instruction, closed loop).

### Language conditioning CONFIRMED (2026-09-15)

Fed the checkpoint the SAME observation (images + state from dataset traj 0) with
two different instructions and compared the predicted first-step action (16-dim):

- `"Pick up the red cup on the table."` → mean|action| 0.17
- `"Place the red wooden block into the yellow box."` → mean|action| 0.45
- **mean abs diff between the two: 0.38** → the policy LISTENS to the language;
  changing only the instruction string changes the action substantially.

So "language mode" works: the same policy does different things for different
instructions. This is the green light to build the mjlab bridge.

**API gotchas hit while writing the test:**
- `LeRobotEpisodeLoader(dataset_path=..., modality_configs=policy.get_modality_config())`
  — no `embodiment_tag` kwarg.
- Build obs as flat keys `state.<k>` / `video.<k>` (np arrays) plus the language
  key set to a **plain string** (not a list/ndarray — `parse_observation_gr00t`
  wraps a str itself; a list hits `arr[None,:]` and throws).
- `get_action` returns `(chunk, _)`; `chunk[k]` is `(batch, horizon, dim)`. Mirror
  eval's `parse_action_gr00t`: `{f"action.{k}": chunk[k][0]}`, then index `[j]`
  for horizon step j.

Status: language conditioning verified — proceed to build the mjlab Dex1 env.

## Why GR00T (and why not RL / not UnifoLM)

- **RL grasp from scratch failed.** Both `Mjlab-Lift-Cube-G1` (success ~5%) and
  `Mjlab-Place-Box-G1` (success 0% at 740 iters, `position_error` flat) plateaued.
  PPO grasp exploration with the Dex3 43-DOF hand from random init is too hard for
  the current reward stack. Decision: switch to imitation/VLA, which learns from
  the 216 demonstrations that already grasp successfully.
- **GR00T N1.7 chosen over UnifoLM-VLA (Unitree).** UnifoLM is more G1/dataset
  native, but pins `torch==2.5.1` + `flash-attn==2.5.6`, neither of which supports
  Blackwell (sm_121); porting it to GB10 is heavy CUDA work. GR00T officially
  supports DGX Spark (GB10) with prebuilt flash-attn/torchcodec wheels.
- **Mac is dev-only.** flash-attn is CUDA-only (no MPS), so the M1 Pro (16 GB)
  cannot run GR00T; all training and inference happen on the Spark. Mac is used
  for editing configs and preparing data.

## Hardware / platform

- DGX Spark: ARM aarch64 + GB10 Blackwell GPU, **compute capability (12, 1) =
  sm_121**. CUDA 13.0 driver. Repo at `/home/nexpie/workspace/anun/Isaac-GR00T`.
- SSH/credentials in `ssh-spark.txt` (gitignored). See [dgx-spark-gb10-setup] memory.

## What was done (chronological)

1. **Cloned Isaac-GR00T** (`git clone`) to the Spark workspace. Latest is N1.7.
2. **Ran the Spark install** (`scripts/deployment/spark/install_deps.sh`) — needs
   `sudo` for system packages (NVPL libs, ffmpeg, build tools), so the **user runs
   it manually on the Spark terminal** (SSH non-interactive can't enter the sudo
   password; do NOT set passwordless sudo — leaves a security hole).
   - Hit a broken apt repo: `/etc/apt/sources.list.d/ookla_speedtest-cli.list`
     returned no Release file, blocking `apt-get update`. Fix: rename it to
     `.disabled`, then `apt-get install -y ffmpeg`.
   - **ffmpeg 6** installed — within torchcodec's required range (FFmpeg 4-7).
   - flash-attn installed from the **prebuilt aarch64 wheel** shipped in the repo
     (`scripts/deployment/spark/wheels/flash_attn-2.8.3-cp312-cp312-linux_aarch64.whl`).
   - Also run `scripts/patch_triton_cuda13.sh` and
     `source scripts/activate_spark.sh` (sets CUDA_HOME=/usr/local/cuda-13.0,
     TRITON_PTXAS_PATH). Verified `import gr00t` → OK.
3. **Downloaded the dataset** `unitreerobotics/G1_Dex1_PickPlaceRedBlock_Dataset_Sim`
   to `~/workspace/anun/datasets/G1_PickPlaceRedBlock` (216 episodes, 2 tasks:
   "Pick up the red cup" and "Place the red wooden block into the yellow box").
   - The dataset is **codebase_version v3.0** (not v2.1 as first assumed).
4. **Converted v3.0 → v2.1** with GR00T's own
   `scripts/lerobot_conversion/convert_v3_to_v2.py` (needs its own uv venv +
   lerobot pinned commit). Output landed under a nested path (converter appends
   the repo-id to `--root`):
   `datasets/G1_PickPlaceRedBlock/unitreerobotics/G1_Dex1_PickPlaceRedBlock_Dataset_Sim`.
   Result: `meta/tasks.jsonl`, `episodes.jsonl`, `episodes_stats.jsonl`, 216
   `data/chunk-000/episode_*.parquet`, `codebase_version: v2.1`. Verified.
5. **Wrote `meta/modality.json`** for the 16-dim state/action layout:
   `left_arm [0:7]`, `right_arm [7:14]`, `left_gripper [14:15]`,
   `right_gripper [15:16]`; three cameras
   (`cam_high`=cam_left_high, `cam_left_wrist`, `cam_right_wrist`);
   annotation `human.task_description` ← `task_index`.
6. **Wrote the modality config** `examples/G1Dex1/g1_dex1_config.py` — registers
   under `EmbodimentTag.NEW_EMBODIMENT` (the dataset is upper-body + Dex1 gripper,
   which does NOT match the `UNITREE_G1` preset = full-body with legs/waist). Arms
   use RELATIVE action rep, grippers ABSOLUTE; action horizon 16 steps.
7. **HuggingFace gated access.** GR00T's VLM backbone `nvidia/Cosmos-Reason2-2B`
   is a gated repo. The user requested access and ran `hf auth login` on the Spark
   (user: anunpanya). Verified gated download works (401 gone). HF is **free** —
   "gated" means request-access, not paid.
8. **10-step fine-tune smoke test** via
   `gr00t/experiment/launch_finetune.py --base-model-path nvidia/GR00T-N1.7-3B
   --dataset-path <v2.1 path> --embodiment-tag NEW_EMBODIMENT
   --modality-config-path examples/G1Dex1/g1_dex1_config.py ...`.

## STATUS: smoke test PASSED (2026-09-14) — blocker resolved

The 10-step fine-tune smoke test succeeded end-to-end with the final cu13/torch2.10
stack: **0 nvrtc errors**, all 10 steps completed (`train_loss=1.1123`,
grad_norm≈0.38), and `checkpoint-10` (+ processor + experiment_cfg) was saved to
`~/workspace/anun/vla_ckpt_smoke`. No modality/shape errors — the dataset (v2.1 +
modality.json) and `g1_dex1_config.py` (NEW_EMBODIMENT, 16-dim) are correct, and
flash-attn runs ("Casting fp32 back to bf16 for flash-attn compatibility").
The pipeline is ready for a full fine-tune run.

### CAUTION: dataloader workers hang the Spark

A first full run with `--dataloader-num-workers 4` drove the Spark into
swap/hang (had to hard reboot): step 1 took **314s** (vs 2.6s in the smoke test)
and memory blew up. Root cause: 4 dataloader workers each decode 3-camera video
via torchcodec and prefetch multiple batches, exhausting the 122 GB unified
memory. A controlled 5-step test with `--dataloader-num-workers 0` ran at
**5.09s/step**, `train_loss=1.13`, with free memory never dropping below ~13 GB.

**Rule for training on this Spark:** use `--dataloader-num-workers 0` (or at most
1) for this dataset. Consider a memory guard (kill the run if `free -g` avail
drops below ~8 GB) for long runs. A watchdog wrapper is in
`~/workspace/anun/vla_test5_mem.log`'s launcher.

Launch command that works (run with `source scripts/activate_spark.sh` first,
`PYTORCH_JIT=0`, `setsid` to survive SSH drop):
```
gr00t/experiment/launch_finetune.py --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path <v2.1 dataset path> --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/G1Dex1/g1_dex1_config.py --num-gpus 1 \
  --output-dir <out> --save-steps N --max-steps N --global-batch-size 8 \
  --dataloader-num-workers 2
```

## FULL TRAINING COMPLETE (2026-09-14) — first G1 VLA model

The 2000-step fine-tune finished cleanly in **58m37s**:
- loss **1.1 → ~0.11** (10x drop), cosine LR schedule ran to ~0.
- `--dataloader-num-workers 0` + memory guard: min free memory **14 GB**, guard
  never tripped, no hang/reboot.
- checkpoints saved at 500/1000/1500/**2000** in
  `~/workspace/anun/vla_ckpt_g1placebox/checkpoint-2000` (final).

This is the first fine-tuned GR00T N1.7 VLA for the G1 pick-and-place task.

### Open-loop eval PASSED (2026-09-15)

Evaluated checkpoint-2000 on 3 trajectories (150 steps each): **avg MSE 0.21,
MAE 0.17** (unnormalized joint-space action). The per-dimension plot
(`~/workspace/anun/vla_eval_plot.png`) shows predicted actions tracking the
ground-truth trajectory shape for both arms (dims 0-13), and — importantly —
the model captures the **left-gripper open/close step** (dim 14 jumps ~0→5 at
step ~60). Right gripper (dim 15) stays near 0, matched. So the policy learned
the pick-place motion, not just minimized loss. MAE ≈ 0.17 rad (~10°) is a
reasonable starting point for 2000 steps / 216 demos; more steps or demos would
tighten it.

**Eval gotcha:** `open_loop_eval.py` has **no** `--modality-config-path` arg (unlike
`launch_finetune.py`). It reads modality from the dataset's `meta/modality.json`,
and the NEW_EMBODIMENT modality config must be registered by importing
`g1_dex1_config` first. Run via `PYTHONPATH=examples/G1Dex1 python -c "import
g1_dex1_config; sys.argv=[...]; exec(open('gr00t/eval/open_loop_eval.py').read())"`.
Args: `--model-path <ckpt> --dataset-path <v2.1> --embodiment-tag new_embodiment
--traj-ids 0 1 2 --steps 150 --save-plot-path <png>`.

### (earlier) open-loop eval setup notes `gr00t/eval/open_loop_eval.py` CLI takes
`--model-path <checkpoint-2000>`, `--dataset-path <v2.1>`,
`--embodiment-tag new_embodiment`, `--modality-config-path examples/G1Dex1/g1_dex1_config.py`,
`--traj-ids 0 1 ...`, `--steps N`, `--save-plot-path <png>`, `--denoising-steps 4`.
NOTE: it defaults to host/port 5555 (client/server) — check whether passing
`--model-path` runs the policy in-process, or whether a policy server
(`scripts/inference_service.py` or similar) must be started first. Run with
workers=0 + memory guard, and `source scripts/activate_spark.sh` + `PYTORCH_JIT=0`
+ LD not needed (flash-attn is cu13). Final checkpoint:
`~/workspace/anun/vla_ckpt_g1placebox/checkpoint-2000`.

## (history) BLOCKER that was resolved: nvrtc sm_121 during training

The smoke test fails with:

```
nvrtc: error: invalid value for --gpu-architecture (-arch)
```

- Origin: Qwen3-VL vision backbone `rot_pos_emb`
  (`transformers/models/qwen3_vl/modeling_qwen3_vl.py:610`) via PyTorch's
  **jiterator** (runtime elementwise-kernel compilation), NOT flash-attn.
- **Root cause found:** the installed torch is **`2.9.0+cu128` (CUDA 12.8)**. Its
  bundled nvrtc supports archs up to **sm_120** only; the GB10 is **sm_121**, so
  jiterator asks nvrtc for `sm_121` and it rejects it. This is the SAME class of
  Blackwell problem hit in mjlab, but a deeper code path.
- `PYTORCH_JIT=0` (the mjlab fix) does **not** help — jiterator is below that flag.
- `TORCH_CUDA_ARCH_LIST=12.0` does **not** help — it only affects extension builds.
- `LD_PRELOAD` of the system CUDA-13.0 nvrtc (`/usr/local/cuda-13.0/lib64/libnvrtc.so.13`,
  which DOES know sm_121: archs list includes 121) does **not** help — torch
  `_preload_cuda_deps` loads the bundled `nvidia/cuda_nvrtc/lib/libnvrtc-builtins.so.12.8`
  at import, before the preload can override it.

### The fix (per Isaac-GR00T issue #474)

NVIDIA/Isaac-GR00T issue #474 ("Running GR00T Fine-Tuning on DGX Spark") confirms
the fix: install **torch cu130** (`torch 2.9.1+cu130` from
`https://download.pytorch.org/whl/cu130`), whose nvrtc knows sm_121. The existing
flash-attn 2.8.3 aarch64 wheel works with cu130. Use `--no-deps` when reinstalling
GR00T so torch is not downgraded back to cu128.

**Rollback reference** (GR00T venv before the cu130 change, 2026-09-14):
`torch 2.9.0+cu128, torchvision 0.24.0, nvidia-cuda-nvrtc-cu12 12.8.93,
flash-attn 2.8.3, torchcodec 0.8.0, triton 3.5.0`. If cu130 breaks something,
reinstall these exact versions from the pytorch-cu128 index.

### The real fix direction (superseded by the above)

The Spark install script (`install_deps.sh:112`) targets
`spark-cu130-torch2.10` / arch `121` — i.e. the intended stack is **torch 2.10 +
CUDA 13.0**, which knows sm_121. pyproject pins `torch==2.9.0` with a comment
(line 54): *"cu12 wheels only exist for x86_64; Blackwell / aarch64 requires
cu13."* So the env resolved the WRONG torch variant (cu128 instead of cu13x).

Next step: install the CUDA-13 / Blackwell-aware torch build the Spark path
expects (torch 2.10 cu130, or the aarch64 cu13 wheel), instead of 2.9.0+cu128,
then re-run the smoke test. This changes ONLY the GR00T venv
(`Isaac-GR00T/.venv`), which is isolated — do not touch system CUDA or other
services (user asked to be careful about other running services).

### Applied fix (2026-09-14) — torch cu130

- `uv pip install --index-url .../cu130` did **not** work: `[tool.uv.sources]` in
  the Spark pyproject pins torch to the `pytorch-cu128` index and overrides
  `--index-url`. Had to install from the **explicit wheel URL** with `--no-deps`:
  `torch @ https://download.pytorch.org/whl/cu130/torch-2.9.1%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl`.
  → torch is now **2.9.1+cu130**; jiterator test (`torch.special.i0` on cuda)
  passes with **no nvrtc error** (torch loads nvrtc 13 from its bundled
  `nvidia/cu13/lib/libnvrtc.so.13`). The "max sm_120" warning still prints but is
  harmless — nvrtc 13 JITs sm_121 fine.
- Switching torch to cu130 broke the two cu12-built packages (expected):
  - **flash-attn** (`libcudart.so.12 not found`): fixed WITHOUT rebuild by putting
    the still-present cu12 cudart on the loader path at run time:
    `LD_LIBRARY_PATH=<venv>/lib/python3.12/site-packages/nvidia/cuda_runtime/lib`.
    flash-attn 2.8.3 then imports fine.
  - **torchvision** (`operator torchvision::nms does not exist` = ABI mismatch):
    reinstalled **torchvision 0.25.0+cu130** from the explicit cu130 wheel with
    `--no-deps` (cu130 has no 0.24.x; 0.25.0 is the lowest, pairs OK with torch
    2.9.1). torchcodec 0.10.0 from the repo wheel is next to verify.
- Training must run with the LD_LIBRARY_PATH above so flash-attn resolves cudart 12.

### FINAL working stack (2026-09-14) — all cu13/torch2.10, matched

The 2.9.1 path snowballed: torchvision cu130 has no 0.24.x (needs `torch==2.10.0`),
and flash-attn's cu12torch2.9 wheel breaks on torch 2.10 (C++ ABI:
`undefined symbol: c10_cuda_check_implementation`). The clean solution is to move
the WHOLE stack to torch 2.10 + cu13, for which matched wheels exist:

| package | version | source |
| --- | --- | --- |
| torch | **2.10.0+cu130** | `download.pytorch.org/whl/cu130/torch-2.10.0%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl` |
| torchvision | **0.25.0+cu130** | `download.pytorch.org/whl/cu130/torchvision-0.25.0%2Bcu130-cp312-cp312-manylinux_2_28_aarch64.whl` |
| flash-attn | **2.8.3+cu13torch2.10** | Dao-AILab release v2.8.3, `flash_attn-2.8.3%2Bcu13torch2.10cxx11abiTRUE-cp312-cp312-linux_aarch64.whl` |
| torchcodec | **0.10.0a0** | repo wheel `scripts/deployment/spark/wheels/torchcodec-0.10.0a0-cp312-cp312-linux_aarch64.whl` |

Each installed with `uv pip install --python <venv>/bin/python --no-deps [--reinstall-package <name>] "<name> @ <url-or-path>"`.
All four import cleanly and pass: `flash_attn_2_cuda` load, `torchcodec VideoDecoder`,
`torchvision.ops.nms`, and the jiterator (`torch.special.i0` on cuda) with NO nvrtc
error. With the cu13 flash-attn, **the LD_LIBRARY_PATH cudart-12 hack is no longer
needed** — flash-attn finds cudart 13 on its own.

**Rollback:** the original cu128 stack is in the reference above; reinstall those
exact versions from the pytorch-cu128 index to revert.

## Safety notes

- All fixes must be scoped to the GR00T venv / per-process env vars. Do NOT modify
  system CUDA, global libs, or use passwordless sudo. The user explicitly asked to
  avoid impacting other services on the Spark.

## Key paths

- GR00T repo: `/home/nexpie/workspace/anun/Isaac-GR00T`
- venv: `Isaac-GR00T/.venv` (Python 3.12)
- dataset (v2.1): `~/workspace/anun/datasets/G1_PickPlaceRedBlock/unitreerobotics/G1_Dex1_PickPlaceRedBlock_Dataset_Sim`
- modality config: `Isaac-GR00T/examples/G1Dex1/g1_dex1_config.py`
- smoke-test logs: `~/workspace/anun/vla_smoke.log`, `vla_smoke2.log`
- launch: `source scripts/activate_spark.sh` then
  `gr00t/experiment/launch_finetune.py` (setsid to survive SSH drop)
