# VLA bridge — GR00T policy สำหรับ G1 หยิบของวางกล่อง

โฟลเดอร์นี้เก็บไฟล์ที่เกี่ยวกับการนำโมเดล VLA (NVIDIA Isaac GR00T N1.7 ที่ fine-tune
แล้ว) มาใช้กับหุ่น Unitree G1 — ตั้งแต่ทดสอบคำสั่ง ไปจนถึง (เป้าหมายถัดไป) การ
เชื่อมเข้า mjlab sim

รายละเอียดเทคนิคการติดตั้ง/เทรน/แก้ปัญหา (CUDA 13 / sm_121, dependency wheels ฯลฯ)
อยู่ที่ `docs/development/g1_groot_vla_setup.md` และคำอธิบายหลักการภาษาไทยอยู่ที่
`docs/development/vla_หลักการทำงาน_TH.md`

## ไฟล์ในโฟลเดอร์

| ไฟล์ | คืออะไร |
| --- | --- |
| `g1_dex1_config.py` | modality config: บอก GR00T ว่า state/action 16-dim map ยังไง (แขนซ้าย/ขวา 7+7 + gripper 2) + กล้อง 3 ตัว ใช้ tag `NEW_EMBODIMENT` |
| `modality.json` | ไฟล์เดียวกันในรูป JSON ที่วางใน `meta/` ของ dataset |
| `test_vla_commands.py` | ทดสอบว่า policy ฟังคำสั่งไหม + วัดความแม่นยำ (open-loop) |

## รันบน DGX Spark

โมเดล GR00T รันได้เฉพาะบนเครื่องที่มี NVIDIA GPU (flash-attn ต้องใช้ CUDA) —
MacBook รันไม่ได้ ใช้ Spark:

```bash
cd ~/workspace/anun/Isaac-GR00T
source scripts/activate_spark.sh          # ตั้ง CUDA 13 env
PYTORCH_JIT=0 PYTHONPATH=examples/G1Dex1 \
  uv run --no-sync python examples/G1Dex1/test_vla_commands.py \
    --model-path ~/workspace/anun/vla_ckpt_g1placebox/checkpoint-2000 \
    --dataset-path ~/workspace/anun/datasets/G1_PickPlaceRedBlock/unitreerobotics/G1_Dex1_PickPlaceRedBlock_Dataset_Sim
```

## test_vla_commands.py ทดสอบอะไร

ใช้ observation จริงจาก dataset (open-loop) วัด 3 อย่างที่อธิบายได้:

1. **ฟังคำสั่งไหม** — ป้อน obs เดียวกัน + คำสั่งต่างกัน แล้วดูว่า action ต่างกันไหม
   (ต่าง = policy อ่านภาษา ไม่ได้เมิน)
2. **แม่นแค่ไหน** — action ที่ทำนาย เทียบ action จริงในเดโม (MAE เป็นองศา/ข้อต่อ)
3. **คำสั่งถูก vs ผิด** — ใส่คำสั่งตรง task ควรแม่นกว่าใส่คำสั่งผิด task

**สำคัญ:** ทดสอบวัดที่ **จังหวะกลาง (step 300)** ไม่ใช่จังหวะแรก — เพราะทั้งสอง
task เริ่มจากท่าพักเหมือนกัน จังหวะแรกเลยดูคล้ายกัน ความต่างของ task ชัดตอนกลาง
(ตอนแขนเลือกทิศทางเอื้อม)

### ผลล่าสุด (checkpoint-2000, step 300)

```
1. ฟังคำสั่ง:  action ต่างกัน 0.133 rad (7.6°) เมื่อคำสั่งต่าง  -> ฟังจริง
2. ความแม่น:   MAE เฉลี่ย 0.063 rad = 3.6°/ข้อต่อ            -> ทำนายใกล้เดโม
3. คำสั่งถูก (4.7°) < คำสั่งผิด (5.0°)                        -> policy ใช้คำสั่งจริง
```

open-loop วัดว่า policy ทำนายใกล้คนไหม ยังไม่ใช่การพิสูจน์ว่าหยิบสำเร็จจริง —
การพิสูจน์นั้นต้อง closed-loop ใน sim/หุ่นจริง (ขั้นถัดไป)

## ไฟล์/script ที่ใช้เทรนโมเดล VLA

**script เทรนหลักเป็นของ GR00T (ไม่ได้เขียนเอง)** อยู่บน Spark ที่
`~/workspace/anun/Isaac-GR00T/gr00t/experiment/launch_finetune.py` — เราแค่เรียก
ใช้พร้อม arguments โดยชี้ไปที่ dataset + config ของเรา:

```bash
cd ~/workspace/anun/Isaac-GR00T
source scripts/activate_spark.sh
PYTORCH_JIT=0 CUDA_VISIBLE_DEVICES=0 NUM_GPUS=1 \
  uv run --no-sync python gr00t/experiment/launch_finetune.py \
    --base-model-path nvidia/GR00T-N1.7-3B \
    --dataset-path ~/workspace/anun/datasets/G1_PickPlaceRedBlock/unitreerobotics/G1_Dex1_PickPlaceRedBlock_Dataset_Sim \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/G1Dex1/g1_dex1_config.py \
    --num-gpus 1 --output-dir ~/workspace/anun/vla_ckpt_g1placebox \
    --save-steps 500 --save-total-limit 5 --max-steps 2000 \
    --global-batch-size 8 --dataloader-num-workers 0
```

- **ไฟล์ที่เราเขียนเอง** (บอกว่าเทรนกับหุ่น/ข้อมูลแบบไหน) = `g1_dex1_config.py`
  ในโฟลเดอร์นี้ (สำเนาอยู่บน Spark ที่ `examples/G1Dex1/`) และ `modality.json`
  ที่วางใน `meta/` ของ dataset
- **`--dataloader-num-workers 0` จำเป็น** — ถ้าใส่มากกว่านี้ Spark จะค้างจนต้อง
  reboot (worker หลายตัว decode video 3 กล้องพร้อมกันจนกิน memory หมด)
- **โมเดลรันได้เฉพาะบน Spark** (flash-attn ต้องใช้ CUDA — MacBook รันไม่ได้)
- ผลเทรน: 2000 steps ~58 นาที, loss 1.1 → 0.11, ได้ `checkpoint-2000`

รายละเอียดการติดตั้ง stack (CUDA 13 / sm_121 wheels) และปัญหาที่เจอ อยู่ใน
`docs/development/g1_groot_vla_setup.md`

## เชื่อมเข้า mjlab sim — closed-loop (ทำได้แล้ว)

VLA คุมหุ่น G1 ใน mjlab sim ได้จริงแบบ closed-loop แล้ว ทั้ง sim และ policy รันบน
Spark คนละ venv (torch คนละ build) คุยกันผ่าน localhost:

```
mjlab venv ── vla_mjlab_client.py ──ZMQ:5555──► gr00t venv ── run_gr00t_server.py
(torch cu128)                                    (torch cu130, โหลด checkpoint)
```

- `src/mjlab/asset_zoo/robots/unitree_g1/g1_gripper_constants.py` — G1 +
  parallel-jaw gripper (action 16-dim ตรง dataset)
- `src/mjlab/tasks/manipulation/config/g1_vla/` — env: หุ่น+โต๊ะ+cube+box + กล้อง 3 ตัว
- `vla_bridge/vla_mjlab_client.py` — client: render 3 กล้อง + อ่าน state → ส่ง server
  → remap action → step → save video

**วิธีรัน (2 terminal บน Spark):**
```bash
# terminal 1 — policy server (gr00t venv)
cd ~/workspace/anun/Isaac-GR00T && source scripts/activate_spark.sh
PYTORCH_JIT=0 PYTHONPATH=examples/G1Dex1 uv run --no-sync python \
  gr00t/eval/run_gr00t_server.py --model-path ~/workspace/anun/vla_ckpt_g1placebox/checkpoint-2000 \
  --embodiment-tag new_embodiment --modality-config-path examples/G1Dex1/g1_dex1_config.py --port 5555

# terminal 2 — mjlab client (mjlab venv)
cd ~/workspace/anun/mjlab-custom
MUJOCO_GL=egl PYTORCH_JIT=0 uv run --no-sync python vla_bridge/vla_mjlab_client.py \
  --instruction "Place the red wooden block into the yellow box." \
  --steps 60 --video-path ~/workspace/anun/vla_rollout.mp4
```

**สองโหมด (policy ตัวเดียว):** เปลี่ยน `--instruction` = โหมดสั่งด้วยคำสั่ง;
instruction คงที่ = โหมดอัตโนมัติ (closed loop จนจบ)

**ผลปัจจุบัน:** pipeline ทำงาน end-to-end — หุ่นขยับตาม VLA และบันทึกวิดีโอได้
การหยิบยังไม่สำเร็จ (โมเดลเทรน 2000 steps + sim-to-real gap ระหว่าง Unitree sim
ที่เก็บ data กับ mjlab + gripper เป็น stand-in) เป้าหมายขั้นนี้คือพิสูจน์ว่า bridge
ทำงานครบวงจร ไม่ใช่ความสมบูรณ์แบบของการหยิบ

## หมายเหตุ: ลำดับ action (sim ≠ dataset)

mjlab จัด action targets ตามข้าง → ลำดับ sim = `[แขนซ้าย 7, gripper ซ้าย, แขนขวา 7,
gripper ขวา]` แต่ dataset = `[แขนซ้าย 7, แขนขวา 7, gripper ซ้าย, gripper ขวา]`
client จึง remap ก่อนสั่ง (`ds_to_action` ใน vla_mjlab_client.py)

## ต่อยอดสู่หุ่นจริง (ROS2)

สถาปัตยกรรม client-server ทำให้ต่อยอดง่าย: **policy server reuse ได้เลย** เปลี่ยน
แค่ client — เขียน ROS2 node ที่ subscribe กล้อง + `/joint_states`, เรียก
`RemotePolicy` (คลาสในไฟล์นี้ reuse ได้), publish action ไป joint controller.
ต้องจัดการเพิ่ม: sim-to-real gap (อาจ fine-tune ด้วย demos จริง), safety limits,
real-time 30Hz sync, และ map เป็น Dex1 gripper จริง
