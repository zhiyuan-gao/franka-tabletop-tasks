# Franka Tabletop Tasks: Training, Checkpoints, Evaluation, and Sim-to-Real Handoff

**Repository:** `/raid/home/than/zhiyuan/ICRA2027/franka-tabletop-tasks`  
**Final snapshot:** `/raid/home/than/zhiyuan/ICRA2027/franka-tabletop-tasks/Final-training`  
**Environment:** `lw_benchhub`  
**Robot target:** Franka Panda using ROS 1, `franka_ros`/`libfranka`, a Cartesian/IK command bridge, and a fixed external RGB camera  
**Tasks:** `push_cube_to_region`, `drawer_opening`, `object_to_drawer`, and `nut_pouring`

This document describes the final single-task and multitask behavior-cloning (BC) and residual-RL pipelines, the exact file and checkpoint conventions, simulator evaluation, and the work required to execute the policies safely on the real robot. It intentionally does not report experimental performance numbers.

## 1. What is ready and what is not

| Component | State | Used on the real robot? |
|---|---|---|
| Dataset reader, augmentation, action normalization | Implemented | Preprocessing contract must be reproduced |
| Single-task chunked BC | Implemented | Yes |
| Single-task residual TD3+BC actor | Implemented | Yes, actor only |
| Multitask chunked BC | Implemented | Yes |
| Multitask residual actor | Implemented | Yes, actor only |
| Isaac Lab task environments, rewards, oracle observations, critics | Implemented | No |
| Simulator checkpoint selection and video evaluation | Implemented | No, simulator only |
| ROS 1 camera/joint-state/TF adapter | **Not yet implemented** | Required |
| Real Cartesian safety filter and controller bridge | **Not yet implemented** | Required |
| Real trial logger and independent outcome evaluator | **Not yet implemented** | Required |

The learned actor does not require simulator state. At deployment it consumes RGB, proprioception, and a task selection. The RL critic, simulator oracle observation, reward function, and simulator termination manager are training/evaluation components and must not be required by the real policy node.

## 2. Activate the environment

Run this at the start of every server session:

```bash
source /raid/home/than/miniforge3/etc/profile.d/conda.sh
conda activate lw_benchhub

export ISAAC_PYTHON=/raid/home/than/miniforge3/envs/lw_benchhub/bin/python
cd /raid/home/than/zhiyuan/ICRA2027/franka-tabletop-tasks
```

To expose only physical GPU 1:

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
```

After this mapping, use `--device cuda:0`: CUDA logical device 0 is physical GPU 1. Isaac/Omniverse may still display both physical adapters in its startup table. Confirm the actual process placement with `nvidia-smi`.

Do not set `CUDA_VISIBLE_DEVICES` after Isaac Sim has already started. Start a new process whenever the GPU mapping changes.

## 3. Repository inventory

### 3.1 Shared single-task V2 code

| File | Purpose |
|---|---|
| `scripts/task_pipeline_common.py` | Dataset discovery, task validation, simulator observation helpers |
| `scripts/task_pipeline_v2.py` | V2 dataset, action normalization, chunked BC, temporal runner, replay, residual actor/critics, video encoder |
| `scripts/train_task_bc_v2.py` | Generic single-task BC trainer selected by `--task` |
| `scripts/eval_task_bc_v2.py` | Single-task BC rollout, checkpoint sweep/selection, JSON output, MP4 recording |
| `scripts/train_task_residual_v2.py` | Generic single-task residual TD3+BC trainer |
| `scripts/eval_task_residual_v2.py` | Single-task BC+residual evaluation and optional BC comparison |
| `scripts/smoke_test_v2.py` | CPU mock-environment integration and invariants test |

The same files are expected in `Final-training/scripts/`.

### 3.2 Multitask BC code

| File | Purpose |
|---|---|
| `scripts/task_pipeline_multitask_v2.py` | Shared multitask policy, task embeddings, task-specific heads/normalizers, checkpoint loading |
| `scripts/train_multitask_bc_v2.py` | Four-task BC training |
| `scripts/eval_multitask_bc_v2.py` | Multitask BC checkpoint selection/evaluation |

### 3.3 Multitask residual-RL code

The source bundle is:

```text
multitask_residual_rl_v2/
├── bc_checkpoint_adapter_v2.py
├── train_multitask_residual_phase_v2.py
├── eval_multitask_residual_v2.py
├── run_multitask_residual_v2.sh
├── select_multitask_residual_v2.sh
└── eval_multitask_residual_final_v2.sh
```

These files provide a shared residual actor and use separate Isaac processes for heterogeneous tasks. The shell wrappers run task phases, select a residual checkpoint, and run final evaluation.

### 3.4 Environment and calibration code

| Path | Purpose |
|---|---|
| `src/track_b_envs/` | Registered Isaac Lab environments and task definitions |
| `src/track_b_envs/rl/` | Environment creation and normalized action-space helpers |
| `src/track_b_envs/envs/<task>/` | Per-task scene, reward, reset, success, and episode-limit configuration |
| `assets/calibration/background/empty_background.png` | Background used by the simulation camera pipeline |
| `assets/calibration/transforms/T_base_from_camera_usd.npy` | Calibrated camera-to-base transform used by the simulator setup |
| `assets/calibration/transforms/T_base_from_marker.npy` | Marker-to-base calibration artifact |
| `assets/calibration/configs/simulation_camera.yaml` | Camera configuration |

The intended object-to-drawer evaluation horizon was changed to 60 seconds. Verify the active source rather than relying on memory:

```bash
grep -RIn "episode_length_s" \
  src/track_b_envs/envs/object_to_drawer/object_to_drawer_env_cfg.py
```

At a 20 Hz policy rate, 60 seconds corresponds to 1,200 policy steps.

## 4. Verify the final archive before moving it

The earlier `Final-training/scripts/` listing contained only the single-task V2 files. Copy the multitask files before treating that directory as a complete handoff:

```bash
cd /raid/home/than/zhiyuan/ICRA2027/franka-tabletop-tasks

mkdir -p Final-training/scripts

cp -v scripts/task_pipeline_multitask_v2.py Final-training/scripts/
cp -v scripts/train_multitask_bc_v2.py Final-training/scripts/
cp -v scripts/eval_multitask_bc_v2.py Final-training/scripts/

cp -v multitask_residual_rl_v2/bc_checkpoint_adapter_v2.py Final-training/scripts/
cp -v multitask_residual_rl_v2/train_multitask_residual_phase_v2.py Final-training/scripts/
cp -v multitask_residual_rl_v2/eval_multitask_residual_v2.py Final-training/scripts/
cp -v multitask_residual_rl_v2/run_multitask_residual_v2.sh Final-training/scripts/
cp -v multitask_residual_rl_v2/select_multitask_residual_v2.sh Final-training/scripts/
cp -v multitask_residual_rl_v2/eval_multitask_residual_final_v2.sh Final-training/scripts/
```

Check that the snapshot and working scripts are byte-identical:

```bash
for f in \
  task_pipeline_common.py \
  task_pipeline_v2.py \
  train_task_bc_v2.py \
  eval_task_bc_v2.py \
  train_task_residual_v2.py \
  eval_task_residual_v2.py \
  smoke_test_v2.py \
  task_pipeline_multitask_v2.py \
  train_multitask_bc_v2.py \
  eval_multitask_bc_v2.py
do
  cmp "scripts/$f" "Final-training/scripts/$f" || echo "DIFFERENT: $f"
done
```

No output means all compared files match. Also archive the exact `src/track_b_envs/`, `configs/`, calibration assets, `pyproject.toml`, dataset documentation, this handoff, and a checkpoint manifest.

## 5. Dataset contract

The standard task datasets are under:

```text
datasets/primary_50/<task>/
datasets/additional_50_seed20260927/<task>/
```

The additional dataset is used where it was collected, notably for object-to-drawer and nut-pouring. Each episode is an NPZ file with fields equivalent to:

| Key | Type/shape | Meaning |
|---|---|---|
| `observation.rgb` | `uint8 [T,H,W,3]` | RGB policy view |
| `observation.proprio` | `float32 [T,25]` | Robot state and TCP pose |
| `observation.oracle` | task-dependent | Privileged simulator critic input; never a real actor input |
| `action` | `float32 [T,7]` | Six Cartesian delta components and gripper command |
| `reward` | `[T]` | Simulator training signal |
| `terminated`, `truncated`, `success` | `[T]` | Episode labels |
| `timestamp_s` | `[T]` | Control timestamps |

Dataset validation:

```bash
"$ISAAC_PYTHON" scripts/inspect_task_dataset.py \
  --task drawer_opening \
  --dataset-dir "$PWD/datasets/primary_50/drawer_opening"
```

Change only `--task` and the dataset directory for the other tasks. Before future training, inspect every supplied directory and confirm RGB dimensions, 25-dimensional proprioception, seven-dimensional actions, episode counts, timestamp cadence, and task labels.

## 6. Actor input and output contract

### 6.1 Policy input

The actor receives only:

1. A fixed external RGB camera image.
2. A 25-dimensional proprioceptive vector.
3. For the multitask policy, one explicit task identity.

The expected proprioceptive ordering must match `policy_observation()` in `scripts/task_pipeline_common.py`. For the current Franka articulation it is expected to be:

```text
[7 arm joint positions,
 2 finger joint positions,
 7 arm joint velocities,
 2 finger joint velocities,
 TCP x, y, z in the robot base frame,
 TCP quaternion w, x, y, z in the robot base frame]
```

This ordering, units, joint names, TCP frame, base frame, and quaternion convention must be verified directly against the active source and a saved episode. Do not guess them in the ROS node. Isaac uses scalar-first quaternions in this path; the real adapter must convert ROS quaternion ordering to the checkpoint ordering.

### 6.2 Temporal input

The final configuration is stored inside every checkpoint and is the authority. The V2 default temporal contract is:

- policy rate: 20 Hz;
- frame stack: three observations;
- stride: three control steps;
- history buffer length: seven control steps;
- history samples: `t-6`, `t-3`, `t`;
- temporal span: 0.30 seconds;
- action chunk: eight future actions;
- chunk horizon: 0.40 seconds;
- overlapping chunks combined by exponential temporal ensembling.

At episode reset, the current observation fills the entire history and the action ensemble is cleared. Real deployment must reproduce this behavior exactly.

### 6.3 Image preprocessing

The real adapter must reproduce the training/evaluation preprocessing, including quantization order:

1. Acquire RGB, not BGR.
2. Apply the calibrated policy-view crop used by the repository.
3. Resize with bilinear interpolation.
4. Convert through the same `uint8` rounding path used by the dataset and `ChunkRunner`.
5. Convert to `float32` in `[0,1]` only after resizing/rounding.
6. Preserve channel order and stack order.

The simulator camera starts from a 1280×720 calibrated image and produces a square policy view before the network resize. The final multitask run used a high-resolution configuration; load `image_size` and every architecture field from the checkpoint instead of hard-coding them.

The available calibration values and files are useful initialization, but the real camera still requires a fresh verification of intrinsics, distortion, exposure, white balance, focus, crop, and extrinsic alignment.

### 6.4 Policy output

The action has seven dimensions:

```text
[dx, dy, dz, d_axis_angle_x, d_axis_angle_y, d_axis_angle_z, gripper]
```

The first six values are normalized relative Cartesian commands. The last value is binary:

- `+1`: open;
- `-1`: close.

The simulator nominal arm scales are approximately:

```text
translation: [0.05, 0.05, 0.05]
rotation:    [0.08, 0.08, 0.08]
```

These are simulator limits per policy step, not safe real-robot limits. Start real trials with substantially smaller limits and increase only after conformance and safety testing.

The exact translation frame, axis-angle frame, multiplication order, and sign convention must be confirmed from the active Isaac action controller. A frame mismatch can make a visually plausible policy dangerous.

## 7. Single-task BC pipeline

The same trainer is used for all four tasks by changing `--task` and the dataset paths.

Example:

```bash
TASK=drawer_opening
RUN=outputs/${TASK}_bcv2_seed42

"$ISAAC_PYTHON" scripts/train_task_bc_v2.py \
  --task "$TASK" \
  --dataset-dir "$PWD/datasets/primary_50/$TASK" \
  --output-dir "$RUN" \
  --epochs 120 \
  --batch-size 96 \
  --num-workers 4 \
  --amp \
  --seed 42 \
  --device cuda:0 \
  --skip-full-validation \
  2>&1 | tee "train_${TASK}_bcv2_seed42.log"
```

For object-to-drawer or nut-pouring, append the additional dataset:

```bash
  --dataset-dir "$PWD/datasets/additional_50_seed20260927/$TASK"
```

BC features:

- ResNet-style visual encoder with GroupNorm and spatial-softmax keypoints;
- proprioceptive encoder;
- three-frame temporal history;
- eight-step action chunks and ACT-style temporal ensembling;
- per-action-dimension robust normalization stored in the checkpoint;
- independent arm regression and gripper classification heads;
- extra gripper loss around open/close transitions;
- random-shift and photometric augmentation;
- EMA model weights;
- periodic checkpoints for closed-loop selection.

Validation loss is a supervised diagnostic, not the deployment-selection criterion. Select a checkpoint using closed-loop simulator rollouts on a selection seed that is different from the final reporting seed.

```bash
TASK=drawer_opening
RUN=outputs/${TASK}_bcv2_seed42

"$ISAAC_PYTHON" scripts/eval_task_bc_v2.py \
  --task "$TASK" \
  --checkpoint-dir "$RUN/checkpoints" \
  --episodes 30 \
  --num-envs 8 \
  --video-episodes 0 \
  --seed 4242 \
  --select-into "$RUN/bc_selected.pt" \
  --output "$RUN/bc_selection_seed4242.json" \
  --headless \
  --device cuda:0 \
  2>&1 | tee "eval_${TASK}_bcv2_selection_seed4242.log"
```

Final simulator evaluation of only the selected checkpoint, with a video for every trial:

```bash
TASK=drawer_opening
RUN=outputs/${TASK}_bcv2_seed42

"$ISAAC_PYTHON" scripts/eval_task_bc_v2.py \
  --task "$TASK" \
  --checkpoint "$RUN/bc_selected.pt" \
  --episodes 20 \
  --num-envs 1 \
  --video-episodes -1 \
  --video-dir "$RUN/final_eval_videos" \
  --output "$RUN/final_eval.json" \
  --seed 9001 \
  --headless \
  --device cuda:0 \
  2>&1 | tee "eval_${TASK}_bcv2_final.log"
```

Using `--num-envs 1` is slow but makes per-trial video review straightforward. The patched evaluator records every trial when `--video-episodes -1` is set. Confirm that the number of MP4 files equals the requested number of episodes.

## 8. Single-task residual RL

Residual RL freezes the selected BC policy and learns a bounded correction. The arm command is composed as:

```text
final_arm = clamp(bc_arm + residual_arm, -1, 1)
```

The gripper remains a discrete BC decision unless a residual score clears the configured authority and flip margin. The residual actor consumes the frozen BC feature, base BC action, and current proprioception. Its critics use privileged simulator state only during training.

Training example:

```bash
TASK=drawer_opening
BC_RUN=outputs/${TASK}_bcv2_seed42
RL_RUN=outputs/${TASK}_residualv2_seed42

"$ISAAC_PYTHON" scripts/train_task_residual_v2.py \
  --task "$TASK" \
  --bc-checkpoint "$BC_RUN/bc_selected.pt" \
  --dataset-dir "$PWD/datasets/primary_50/$TASK" \
  --output-dir "$RL_RUN" \
  --num-envs 8 \
  --transitions 300000 \
  --eval-interval 25000 \
  --eval-episodes 16 \
  --eval-bc-baseline \
  --seed 42 \
  --headless \
  --device cuda:0 \
  2>&1 | tee "train_${TASK}_residualv2_seed42.log"
```

Append any additional dataset used by that task. Do not reuse RL gripper authority blindly across tasks: pushing and drawer manipulation may retain the BC gripper, while pouring may need controlled correction. The deployed authority and flip margin must match the checkpoint/training configuration.

Evaluate one selected residual checkpoint and record all trials:

```bash
TASK=drawer_opening
BC_RUN=outputs/${TASK}_bcv2_seed42
RL_RUN=outputs/${TASK}_residualv2_seed42

"$ISAAC_PYTHON" scripts/eval_task_residual_v2.py \
  --task "$TASK" \
  --bc-checkpoint "$BC_RUN/bc_selected.pt" \
  --residual-checkpoint "$RL_RUN/checkpoints/residual_best.pt" \
  --episodes 20 \
  --num-envs 1 \
  --compare-bc \
  --video-episodes -1 \
  --video-dir "$RL_RUN/final_eval_videos" \
  --output "$RL_RUN/final_eval.json" \
  --seed 9002 \
  --headless \
  --device cuda:0 \
  2>&1 | tee "eval_${TASK}_residualv2_final.log"
```

## 9. Multitask BC pipeline

The multitask BC uses one shared visual/proprioceptive trunk, a learned task embedding, and task-specific arm/gripper heads and action normalizers. Task identity is therefore a required actor input; it is not inferred from the image.

The trained checkpoint series is under:

```text
outputs/multitask_bcv2_final240_highres_seed42/checkpoints/
```

The final epoch checkpoint is:

```text
outputs/multitask_bcv2_final240_highres_seed42/checkpoints/bc_multitask_epoch_0240.pt
```

If a checkpoint sweep completed successfully, the selected copy should be:

```text
outputs/multitask_bcv2_final240_highres_seed42/bc_multitask_selected.pt
```

Do not assume that this selected file exists. The earlier four-task sweep exited after the first heterogeneous environment, so verify it:

```bash
test -f outputs/multitask_bcv2_final240_highres_seed42/bc_multitask_selected.pt \
  && echo "selected checkpoint exists" \
  || echo "selected checkpoint is absent"
```

Isaac Sim does not reliably destroy one heterogeneous task and create the next inside the same Python process. For robust checkpoint selection and evaluation, start a fresh Isaac process per task. The provided evaluator/wrappers should be used only if they already implement this process isolation. Otherwise loop from the shell and invoke one task per process.

Before repeating multitask training, preserve the exact successful command from the training log. CLI options are authoritative in the active scripts:

```bash
"$ISAAC_PYTHON" scripts/train_multitask_bc_v2.py --help
"$ISAAC_PYTHON" scripts/eval_multitask_bc_v2.py --help
```

The final high-resolution checkpoint carries its architecture configuration, including image size, task order, dimensions, and per-task normalizers. Real and simulator loaders must read this metadata; they must not reconstruct it from assumptions.

## 10. Multitask residual RL

The multitask residual system produces one shared actor checkpoint. Training is round-robin across tasks, with each task phase in a fresh Isaac process. Actor parameters continue across phases. Critics may be reinitialized when task oracle dimensions differ; this is acceptable because critics are simulator-only and are not deployed.

Run the checked-in wrapper from the repository root:

```bash
bash multitask_residual_rl_v2/run_multitask_residual_v2.sh \
  2>&1 | tee train_multitask_residualv2.log
```

Inspect the wrapper before rerunning and confirm that its BC checkpoint is the intended multitask checkpoint:

```bash
grep -nE "BC_CHECKPOINT|OUTPUT|TASK|TRANSITION|GRIPPER|AUTHORITY" \
  multitask_residual_rl_v2/run_multitask_residual_v2.sh
```

The multitask residual output root is:

```text
outputs/multitask_residualv2_epoch240_seed42/
```

Expected checkpoint files:

```text
outputs/multitask_residualv2_epoch240_seed42/checkpoints/shared_latest.pt
outputs/multitask_residualv2_epoch240_seed42/checkpoints/shared_round_01.pt
...
outputs/multitask_residualv2_epoch240_seed42/checkpoints/shared_round_06.pt
outputs/multitask_residualv2_epoch240_seed42/checkpoints/shared_final.pt
outputs/multitask_residualv2_epoch240_seed42/checkpoints/shared_best.pt
```

`shared_best.pt` exists only after successful selection. If it is absent, do not silently relabel `shared_final.pt` as selected; either run the selection wrapper or document that the final checkpoint was used without selection.

```bash
bash multitask_residual_rl_v2/select_multitask_residual_v2.sh \
  2>&1 | tee select_multitask_residualv2.log
```

Run final simulator evaluation with the supplied wrapper:

```bash
bash multitask_residual_rl_v2/eval_multitask_residual_final_v2.sh \
  2>&1 | tee eval_multitask_residualv2_final.log
```

The expected selection summary is:

```text
outputs/multitask_residualv2_epoch240_seed42/selection_summary_seed4242.json
```

Final per-task JSON and MP4 output should be placed below a dedicated directory such as:

```text
outputs/multitask_residualv2_epoch240_seed42/final_eval/
```

For archival evaluation, configure the wrapper to launch one process per task, use a held-out trial initialization protocol, and set its video option to record every trial. Never discard failed-trial videos.

## 11. Checkpoint inventory

### 11.1 Known multitask deployment checkpoints

| Policy | Exact path |
|---|---|
| Multitask BC final epoch | `outputs/multitask_bcv2_final240_highres_seed42/checkpoints/bc_multitask_epoch_0240.pt` |
| Multitask BC selected | `outputs/multitask_bcv2_final240_highres_seed42/bc_multitask_selected.pt` if selection completed |
| Multitask residual selected | `outputs/multitask_residualv2_epoch240_seed42/checkpoints/shared_best.pt` |
| Multitask residual final | `outputs/multitask_residualv2_epoch240_seed42/checkpoints/shared_final.pt` |

### 11.2 Known single-task run roots

| Task | BC run root used during this project |
|---|---|
| Drawer opening | `outputs/drawer_opening_bcv2_seed42/` |
| Object to drawer | `outputs/object_to_drawer_bcv2_seed42_cache100/` |
| Nut pouring | `outputs/nut_pouring_bcv2_seed42_cache100/` |
| Push cube to region | `outputs/push_cube_to_region_bcv2_seed42/` or the exact root recorded in its training log |

The deployment BC file inside a successfully selected single-task run is `bc_selected.pt`. Residual runs store `checkpoints/residual_best.pt` and `checkpoints/residual_final.pt`. Because several experimental output names were used, generate the definitive server manifest instead of relying only on this table:

```bash
cd /raid/home/than/zhiyuan/ICRA2027/franka-tabletop-tasks

find outputs -type f \
  \( -name 'bc_selected.pt' \
     -o -name 'bc_epoch_*.pt' \
     -o -name 'bc_multitask_epoch_*.pt' \
     -o -name 'bc_multitask_selected.pt' \
     -o -name 'residual_best.pt' \
     -o -name 'residual_final.pt' \
     -o -name 'shared_best.pt' \
     -o -name 'shared_final.pt' \) \
  -printf '%p\t%s bytes\n' | sort \
  | tee Final-training/CHECKPOINT_MANIFEST.txt
```

Add SHA-256 hashes before copying checkpoints:

```bash
awk -F '\t' '{print $1}' Final-training/CHECKPOINT_MANIFEST.txt \
  | xargs -r sha256sum \
  > Final-training/CHECKPOINT_SHA256.txt
```

Checkpoint files can be large; store them outside Git if necessary, but preserve the manifest, hashes, exact relative paths, training logs, selection JSON, and environment commit.

## 12. Simulator evaluation rules

Use three independent sources of randomness:

- training seed for model fitting;
- selection seed for choosing an epoch/checkpoint;
- held-out final evaluation seed.

Do not select and report on the same simulator initial states. For multitask evaluation, run each task in a new Isaac process. For videos, record every requested trial and encode the task, policy, checkpoint, trial index, and outcome in either the filename or adjacent metadata.

Every evaluation archive should contain:

- checkpoint path and SHA-256;
- Git commit and dirty status;
- environment/task configuration;
- task name and task ID;
- seed and reset parameters;
- episode horizon;
- BC ensemble settings;
- residual authority, limit, and flip margin;
- per-trial result and length;
- every MP4;
- complete console log and Isaac log path.

Headless GLFW warnings are expected on a server without an X display if the environment proceeds to initialization. CUDA/Warp errors are not automatically harmless: confirm that the process is alive, the intended GPU owns its memory, the environment completes setup, and evaluation emits JSON. Always use `set -o pipefail` so `tee` does not hide a Python failure.

## 13. Real-robot system to implement

The current Isaac evaluation scripts cannot be pointed directly at the Franka. Implement a ROS 1 deployment package with these new components:

```text
real_robot/
├── real_robot_policy_node_v2.py       # synchronized observation -> BC/residual inference
├── franka_cartesian_bridge.py         # safe delta pose -> existing IK/joint controller
├── safety_filter.py                   # limits, workspace, freshness, watchdog
├── trial_logger.py                    # RGB/proprio/actions/timestamps/video/outcome
└── configs/
    └── real_robot_policy.yaml         # topics, frames, limits, checkpoints, task
```

These files do not currently exist. Their implementation must import/reuse policy and checkpoint-loading code from the training repository rather than copying the neural architecture.

### 13.1 ROS node inputs

Parameterize at least:

```yaml
task: drawer_opening
rgb_topic: /camera/color/image_raw
camera_info_topic: /camera/color/camera_info
joint_states_topic: /joint_states
base_frame: panda_link0
tcp_frame: panda_hand_tcp
bc_checkpoint: /absolute/path/to/checkpoint.pt
residual_checkpoint: ""
policy_rate_hz: 20.0
```

Use the actual ROS names on the robot. Synchronize image, joint state, and TF/FK data to a common timestamp. Reject stale or out-of-order observations instead of repeating commands indefinitely.

### 13.2 Policy node flow

At every 20 Hz policy tick:

1. Read the newest synchronized RGB image and robot state.
2. Convert the camera image to the exact policy crop, resize, quantization, channel order, and range.
3. Construct the exact 25-dimensional proprio vector.
4. On reset, fill the history with the current observation and clear the temporal ensemble.
5. Select the explicit task ID.
6. Run BC and obtain the base action plus frozen BC feature.
7. If enabled, run the residual actor and compose it with the BC action using the saved task-specific parameters.
8. Pass the six-dimensional Cartesian command through the real safety filter.
9. Convert it to a short-horizon target pose in the verified coordinate convention.
10. Send the target through the existing Cartesian/IK bridge.
11. Send a gripper goal only when the desired binary state changes.
12. Log the complete observation, base action, residual, final action, controller response, and safety status.

The actor must run in `eval()` mode under `torch.inference_mode()`. Do not load or execute the critic on the robot.

### 13.3 Single-task checkpoint loading

For a single-task BC policy, reuse:

```python
from task_pipeline_v2 import ChunkRunner, load_bc_checkpoint_v2

policy, normalizer, payload = load_bc_checkpoint_v2(
    bc_path,
    device,
    expected_task=task,
    prefer_ema=True,
)
```

Construct `ChunkRunner` from `payload["config"]`. Do not hard-code frame stack, stride, chunk length, image size, architecture dimensions, or normalizer values.

For a single-task residual policy:

```python
from task_pipeline_v2 import load_residual_actor_v2

actor, residual_payload = load_residual_actor_v2(
    residual_path,
    device,
    expected_task=task,
)
```

Obtain the referenced BC checkpoint and composition settings from the residual payload, while allowing an explicit absolute BC path after verifying its hash.

### 13.4 Multitask checkpoint loading

Use the loader in `task_pipeline_multitask_v2.py` or `bc_checkpoint_adapter_v2.py`. Bind the requested task name to the exact task index stored in the checkpoint. Never assume alphabetical task order.

For the shared residual checkpoint, load only its actor state and its saved actor configuration. The deployment input remains the task-conditioned frozen BC feature, base action, and proprioception. Verify that the residual checkpoint records or is paired with the exact BC checkpoint hash used during RL training.

### 13.5 Gripper behavior

The BC model has an independent gripper head. Preserve its binary convention exactly. With residual RL:

```text
score = bc_gripper + authority(task) * residual_gripper
if abs(score) > flip_margin(task):
    final_gripper = sign(score)
else:
    final_gripper = bc_gripper
```

The task-specific authority must match simulator evaluation. Do not use an authority designed for nut-pouring on every task. Real `franka_gripper` action calls may block, so trigger open/close only on state transitions and track completion asynchronously. Add force/width/time limits appropriate to the physical objects.

## 14. Sim-to-real calibration and conformance

### 14.1 Camera matching

1. Mount the real camera at the simulated calibrated pose.
2. Recalibrate intrinsics and distortion.
3. Re-estimate `T_base_from_camera` using the marker/calibration board.
4. Recreate the simulator crop on real images.
5. Lock exposure, white balance, gain, and focus for experiments.
6. Capture paired real/sim initial states and compare crop alignment, table edges, cabinet handle, object positions, color, and scale.
7. Overlay projected 3D landmarks on both images and save the verification images.

The real-background compositing used in simulation reduces appearance mismatch but does not establish geometric or photometric equivalence by itself.

### 14.2 Proprioception matching

At the same nominal home pose, save the simulator and real 25-vectors with labels for every dimension. Confirm:

- radians and radians/second;
- finger joint units and ordering;
- joint-name ordering independent of ROS message order;
- TCP definition and tool offset;
- TCP position in `panda_link0`;
- quaternion order and sign handling;
- no normalization applied twice.

Write a unit test that converts a named ROS `JointState` message and TF transform into the checkpoint vector and compares it with a hand-constructed expected vector.

### 14.3 Action conformance test

Before any learned rollout, test one axis at a time from a safe home pose:

1. Apply a very small positive and negative `dx`; measure TCP movement in the base frame.
2. Repeat for `dy` and `dz`.
3. Repeat each axis-angle component with a very small rotation.
4. Compare observed real deltas with simulator deltas from the same normalized command.
5. Verify sign, coordinate frame, rotation composition order, command duration, and scale.

Do not proceed until all six action axes agree. This test is the highest-priority protection against a controller-frame mismatch.

### 14.4 Dynamics and scene matching

Measure and document:

- object geometry, mass, center of mass, friction, and compliance;
- drawer travel, handle pose, friction, and required opening force;
- table and cabinet poses in the robot base frame;
- bowl/container dimensions and fill quantity;
- gripper fingertip geometry, pads, maximum force, and commanded width;
- controller latency, camera latency, and end-to-end inference latency.

Update the simulator calibration/domain randomization where the measured real values lie outside the training range. For nut-pouring in particular, grasp timing, container pose, gripper closure, and contact/friction mismatch should be investigated before expecting residual RL to repair a systematic perception/controller mismatch.

## 15. Real-robot safety requirements

Never send raw network output to joint torque control. The intended path is:

```text
policy action -> bounded Cartesian delta -> workspace/safety filter
              -> desired TCP pose -> IK/controller -> safe joint command
```

Required guards:

- physical emergency stop and operator deadman;
- collision/reflex configuration appropriate for the experiment;
- hard Cartesian workspace box;
- joint-position margins;
- joint velocity, acceleration, and jerk limits;
- Cartesian translation and rotation step limits;
- force/torque stop thresholds;
- command and observation freshness watchdogs;
- invalid image/proprio/action detection;
- stop on NaN/Inf, TF failure, camera loss, controller rejection, or excessive latency;
- automatic stop at the trial horizon;
- safe retreat/home procedure that is not generated by the learned policy.

Start at roughly 10–20% of simulator Cartesian authority, with the gripper disabled. Increase only after observing action-conformance logs. The exact safe values must be approved for the physical setup by the robot operator.

## 16. Real evaluation protocol

Use staged deployment so failures can be localized.

### Stage A: offline parity

- Load the intended checkpoint and print its metadata/hash.
- Feed saved simulator observations through the simulator and real policy entry points.
- Require numerically matching preprocessed tensors and actions.
- Verify history reset and temporal ensembling across a recorded sequence.

### Stage B: shadow mode

- Run the real camera, joint-state, and TF pipeline at 20 Hz.
- Compute and log policy actions without moving the robot.
- Review action scale, continuity, frame convention, gripper switches, and latency.

### Stage C: guarded motion

- Robot at a safe home pose, no task object contact.
- Execute one clipped arm action, then stop.
- Execute a short arm-only rollout, then stop.
- Enable gripper state transitions.
- Run a shortened task trial with an operator holding the deadman.

### Stage D: BC trials

- Evaluate BC alone first.
- Use a fixed, documented reset procedure and randomized but bounded initial states.
- Record every trial, including failures and safety interruptions.

### Stage E: BC plus residual trials

- Use the same initialization schedule as BC-only trials.
- Start the residual with reduced arm authority and task-specific gripper authority.
- Record BC action, residual action, and composed action separately.
- Stop and return to BC if the residual repeatedly hits safety limits or changes the failure mode adversely.

### Stage F: final protocol

- Freeze code, configuration, checkpoints, hashes, calibration, and object setup.
- Use a held-out initial-state schedule not used to tune the real system.
- Randomize trial order across policy/task conditions.
- Keep outcome annotation independent from the policy process.
- Preserve every video and machine-readable trial record.

Seeds do not define real initial states. Use measured object/cabinet placements, a reset jig or fiducials, and a written trial schedule.

## 17. Real outcome measurement

The real actor must not receive outcome labels or privileged object pose. Outcome measurement can use external tracking or human annotation after/during a trial, but it is an evaluation channel only.

Before experiments, copy the exact active task predicates from `src/track_b_envs/envs/<task>/` into a real-evaluation specification. Define measurable tolerances for:

- cube placement relative to the target region;
- drawer displacement;
- object position inside the drawer, release state, and settling;
- transferred nut/container state after pouring.

For object-to-drawer, note that a visually correct drop can still fail the simulator predicate because the object has not settled, the gripper is not sufficiently open, the drawer is not sufficiently open, or the object lies outside a strict internal bound. The real protocol should distinguish task completion from predicate failure and record both the raw measurements and the final binary label.

## 18. Trial logging format

Create one directory per trial:

```text
real_runs/<date>/<policy>/<task>/trial_<index>/
├── metadata.json
├── trajectory.npz
├── camera.mp4
├── events.jsonl
└── outcome.json
```

`metadata.json` should contain:

- code commit and dirty state;
- checkpoint absolute path and SHA-256;
- task name and task index;
- BC-only or BC+residual mode;
- all policy, ensemble, residual, and safety parameters;
- calibration file hashes;
- ROS topic and frame names;
- object/reset configuration;
- operator and run timestamps.

`trajectory.npz` should contain synchronized timestamps, raw/cropped RGB references, the 25-vector, BC action, residual, composed action, clipped/sent command, measured TCP pose, measured joint state, gripper state, controller status, latency, and safety flags.

The video must span the full trial, including the initial scene and final settling interval. Record failures, aborted trials, and safety stops; never save only favorable videos.

## 19. Recommended deployment order

1. Freeze and hash the repository, environment, calibration, and checkpoints.
2. Complete `Final-training/` with the multitask files and this handoff.
3. Build the ROS observation adapter and unit tests.
4. Build the safety-filtered Cartesian/IK bridge.
5. Prove image, proprio, temporal, and action parity offline.
6. Run shadow mode.
7. Run action-axis conformance at reduced authority.
8. Deploy a single-task BC checkpoint on its matching task.
9. Deploy the multitask BC with explicit task selection.
10. Add the single-task residual actor only after its BC baseline is stable on real hardware.
11. Add the multitask shared residual actor last.
12. Freeze the real evaluation protocol and collect all-trial logs/videos.

This order isolates sim-to-real mismatch from multitask interference and residual-policy behavior.

## 20. Final preflight checklist

- [ ] Correct conda environment is active.
- [ ] Intended GPU/process placement is confirmed.
- [ ] Git commit, dirty diff, Python environment, and Isaac versions are archived.
- [ ] `Final-training/` includes single-task and multitask code.
- [ ] Checkpoint manifest and SHA-256 file are complete.
- [ ] The selected checkpoint exists; a final checkpoint is not mislabeled as selected.
- [ ] Checkpoint task list/order and architecture metadata are inspected.
- [ ] Dataset and online preprocessing are identical.
- [ ] Real proprio ordering and quaternion convention are unit-tested.
- [ ] Camera intrinsics/extrinsics and crop are verified with overlays.
- [ ] All six Cartesian action axes pass conformance tests.
- [ ] Policy loop sustains 20 Hz without stale observations.
- [ ] Workspace, joint, velocity, acceleration, force, and watchdog limits are active.
- [ ] Gripper commands are transition-triggered and nonblocking.
- [ ] BC-only is validated before residual execution.
- [ ] Residual authority is task-specific and initially reduced.
- [ ] Outcome measurement is independent from actor inputs.
- [ ] Every real trial saves video and synchronized machine-readable logs.
- [ ] Emergency stop, deadman, and safe recovery have been rehearsed.

## 21. Useful verification commands

```bash
# Show all final training/evaluation scripts.
find scripts multitask_residual_rl_v2 Final-training/scripts \
  -maxdepth 2 -type f \
  \( -name '*.py' -o -name '*.sh' \) -print | sort

# Show checkpoint sizes and modification times.
find outputs -type f -name '*.pt' -printf '%TY-%Tm-%Td %TH:%TM  %12s  %p\n' | sort

# Show evaluation JSON and video files.
find outputs -type f \( -name '*.json' -o -name '*.mp4' \) -print | sort

# Record the repository state.
git rev-parse HEAD | tee Final-training/GIT_COMMIT.txt
git status --short | tee Final-training/GIT_STATUS.txt
git diff --binary > Final-training/WORKTREE.patch

# Record the Python environment.
conda env export --no-builds > Final-training/lw_benchhub_environment.yml

# Validate Python syntax without starting Isaac.
python -m py_compile scripts/*_v2.py
```

The decisive sim-to-real requirement is exact interface parity: the same image tensor, the same 25 values in the same order and frames, the same temporal state, and the same interpretation of each action component. The matched physical scene is valuable, but it does not replace those tests or the real safety layer.
