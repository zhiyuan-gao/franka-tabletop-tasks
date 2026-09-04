# Drawer opening BC → frozen-BC residual TD3+BC

This bundle adds a single-task training pipeline to
`franka-tabletop-tasks`. It uses the released `drawer_opening` dataset directly;
no converted dataset is created.

The training contract is:

```text
RGB + proprio ──> frozen BC ──> base action
                         │
                         └── feature + base action ──> residual TD3 actor

executed action = base action + bounded residual
```

The actor sees only the deployable `observation.rgb` and
`observation.proprio`. The TD3 critics additionally use the simulator-only
`observation.oracle`. BC stays frozen throughout RL. The residual actor retains
a demonstration anchoring loss whose weight decays from 0.80 to 0.20.

## 1. Copy the scripts into the repository

Extract this bundle so these files appear under the existing repository's
`scripts/` directory:

```text
scripts/drawer_pipeline_common.py
scripts/inspect_drawer_dataset.py
scripts/train_drawer_bc.py
scripts/eval_drawer_bc.py
scripts/train_drawer_residual_td3bc.py
scripts/eval_drawer_residual_td3bc.py
```

Then activate the workstation environment:

```bash
source /raid/home/than/miniforge3/etc/profile.d/conda.sh
conda activate lw_benchhub
cd /raid/home/than/zhiyuan/ICRA2027/franka-tabletop-tasks

export ISAAC_PYTHON=/raid/home/than/miniforge3/envs/lw_benchhub/bin/python
export DRAWER_DATASET="$PWD/datasets/primary_50/drawer_opening"
```

## 2. Validate the actual dataset

This reads every episode and checks shapes, dtypes, synchronization, finite
values, action bounds, binary gripper commands, timestamps, and final success:

```bash
"$ISAAC_PYTHON" scripts/inspect_drawer_dataset.py \
  --dataset-dir "$DRAWER_DATASET" \
  | tee drawer_dataset_validation.log
```

Expected high-level result: 50 episodes, task `drawer_opening`, status `valid`.

## 3. Train BC

The train/validation split is by complete episode, not by individual frame.
This prevents frames from one trajectory leaking into both splits.

```bash
"$ISAAC_PYTHON" scripts/train_drawer_bc.py \
  --dataset-dir "$DRAWER_DATASET" \
  --output-dir outputs/drawer_bc_seed42 \
  --epochs 80 \
  --batch-size 64 \
  --seed 42 \
  --device cuda:0 \
  2>&1 | tee train_drawer_bc_seed42.log
```

Best checkpoint:

```text
outputs/drawer_bc_seed42/bc_best.pt
```

## 4. Run BC inference

```bash
"$ISAAC_PYTHON" scripts/eval_drawer_bc.py \
  --checkpoint outputs/drawer_bc_seed42/bc_best.pt \
  --episodes 20 \
  --num-envs 4 \
  --seed 42 \
  --headless \
  --device cuda:0 \
  --output outputs/drawer_bc_seed42/eval_20.json \
  2>&1 | tee eval_drawer_bc_seed42.log
```

Do not start the long RL run until this closed-loop BC evaluation works and has
non-trivial success. A low validation loss alone is not enough.

## 5. Train frozen-BC residual TD3+BC

```bash
"$ISAAC_PYTHON" scripts/train_drawer_residual_td3bc.py \
  --bc-checkpoint outputs/drawer_bc_seed42/bc_best.pt \
  --dataset-dir "$DRAWER_DATASET" \
  --output-dir outputs/drawer_residual_td3bc_seed42 \
  --transitions 140000 \
  --num-envs 4 \
  --seed 42 \
  --headless \
  --device cuda:0 \
  2>&1 | tee train_drawer_residual_td3bc_seed42.log
```

The default residual limits match the CALVIN setup:

```text
arm residual limit       0.10 per dimension
gripper residual limit   2.00
```

Final checkpoint:

```text
outputs/drawer_residual_td3bc_seed42/checkpoints/residual_final.pt
```

## 6. Run BC and residual inference

This evaluates deterministic BC and deterministic trained residual policies.
It does not run a random-residual control.

```bash
"$ISAAC_PYTHON" scripts/eval_drawer_residual_td3bc.py \
  --bc-checkpoint outputs/drawer_bc_seed42/bc_best.pt \
  --residual-checkpoint outputs/drawer_residual_td3bc_seed42/checkpoints/residual_final.pt \
  --policies bc residual \
  --episodes 20 \
  --num-envs 4 \
  --seed 42 \
  --headless \
  --device cuda:0 \
  --output outputs/drawer_residual_td3bc_seed42/eval_20.json \
  2>&1 | tee eval_drawer_residual_td3bc_seed42.log
```

## Quick checks before a long run

Compile all scripts:

```bash
"$ISAAC_PYTHON" -m py_compile \
  scripts/drawer_pipeline_common.py \
  scripts/inspect_drawer_dataset.py \
  scripts/train_drawer_bc.py \
  scripts/eval_drawer_bc.py \
  scripts/train_drawer_residual_td3bc.py \
  scripts/eval_drawer_residual_td3bc.py
```

Run a short BC pipeline check:

```bash
"$ISAAC_PYTHON" scripts/train_drawer_bc.py \
  --dataset-dir "$DRAWER_DATASET" \
  --output-dir outputs/smoke_drawer_bc \
  --epochs 1 \
  --batch-size 64 \
  --seed 42 \
  --device cuda:0
```

Run a short RL integration check after BC exists:

```bash
"$ISAAC_PYTHON" scripts/train_drawer_residual_td3bc.py \
  --bc-checkpoint outputs/drawer_bc_seed42/bc_best.pt \
  --dataset-dir "$DRAWER_DATASET" \
  --output-dir outputs/smoke_drawer_residual \
  --transitions 2100 \
  --num-envs 2 \
  --checkpoint-interval 2000 \
  --log-interval 500 \
  --seed 42 \
  --headless \
  --device cuda:0
```
