# Generic Track B BC → residual TD3+BC pipeline

The same scripts support every registered task through `--task`:

```text
push_object_to_region
push_cube_to_region
drawer_opening
cabinet_opening
object_to_drawer
nut_pouring
```

The dataset directories are discovered automatically from the repository.
For `object_to_drawer` and `nut_pouring`, both released 50-episode roots are
combined automatically when the additional root exists.

## Install

Copy the six Python files from this bundle into the existing repository's
`scripts/` directory. These filenames are new and do not overwrite the earlier
drawer-specific scripts.

## Activate

```bash
source /raid/home/than/miniforge3/etc/profile.d/conda.sh
conda activate lw_benchhub
cd /raid/home/than/zhiyuan/ICRA2027/franka-tabletop-tasks

export CUDA_VISIBLE_DEVICES=1
export ISAAC_PYTHON=/raid/home/than/miniforge3/envs/lw_benchhub/bin/python
```

## Select one task

Only change this value for another task:

```bash
export TASK=nut_pouring
```

## Inspect its dataset

```bash
"$ISAAC_PYTHON" scripts/inspect_task_dataset.py \
  --task "$TASK"
```

## Train BC

```bash
"$ISAAC_PYTHON" scripts/train_task_bc.py \
  --task "$TASK" \
  --epochs 80 \
  --batch-size 64 \
  --num-workers 4 \
  --seed 42 \
  --device cuda:0 \
  --skip-full-validation \
  2>&1 | tee "train_${TASK}_bc_seed42.log"
```

The automatically selected output is:

```text
outputs/<task>_bc_seed42/bc_best.pt
```

## Evaluate BC

```bash
"$ISAAC_PYTHON" scripts/eval_task_bc.py \
  --task "$TASK" \
  --checkpoint "outputs/${TASK}_bc_seed42/bc_best.pt" \
  --episodes 20 \
  --num-envs 4 \
  --seed 42 \
  --headless \
  --device cuda:0 \
  --output "outputs/${TASK}_bc_seed42/eval_20.json" \
  2>&1 | tee "eval_${TASK}_bc_seed42.log"
```

## Train residual TD3+BC

```bash
"$ISAAC_PYTHON" scripts/train_task_residual_td3bc.py \
  --task "$TASK" \
  --bc-checkpoint "outputs/${TASK}_bc_seed42/bc_best.pt" \
  --transitions 140000 \
  --num-envs 4 \
  --seed 42 \
  --headless \
  --device cuda:0 \
  --skip-full-validation \
  2>&1 | tee "train_${TASK}_residual_td3bc_seed42.log"
```

The automatically selected final checkpoint is:

```text
outputs/<task>_residual_td3bc_seed42/checkpoints/residual_final.pt
```

## Evaluate BC and trained residual

```bash
"$ISAAC_PYTHON" scripts/eval_task_residual_td3bc.py \
  --task "$TASK" \
  --bc-checkpoint "outputs/${TASK}_bc_seed42/bc_best.pt" \
  --residual-checkpoint "outputs/${TASK}_residual_td3bc_seed42/checkpoints/residual_final.pt" \
  --policies bc residual \
  --episodes 20 \
  --num-envs 4 \
  --seed 42 \
  --headless \
  --device cuda:0 \
  --output "outputs/${TASK}_residual_td3bc_seed42/eval_20.json" \
  2>&1 | tee "eval_${TASK}_residual_td3bc_seed42.log"
```

Evaluation is deterministic and contains no random-residual policy.

## Compatibility

The generic drawer evaluator accepts the already-trained drawer checkpoint
created by the earlier `train_drawer_bc.py`, so retraining drawer is not needed.
