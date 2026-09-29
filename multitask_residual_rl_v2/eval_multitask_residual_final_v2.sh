#!/usr/bin/env bash
set -euo pipefail

source /raid/home/than/miniforge3/etc/profile.d/conda.sh
conda activate lw_benchhub
cd /raid/home/than/zhiyuan/ICRA2027/franka-tabletop-tasks

export ISAAC_PYTHON=/raid/home/than/miniforge3/envs/lw_benchhub/bin/python
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1

BC="$PWD/outputs/multitask_bcv2_final240_highres_seed42/checkpoints/bc_multitask_epoch_0240.pt"
ROOT="$PWD/outputs/multitask_residualv2_epoch240_seed42"
RL="$ROOT/checkpoints/shared_best.pt"

for TASK in push_cube_to_region drawer_opening object_to_drawer nut_pouring; do
  case "$TASK" in
    push_cube_to_region) SEED=9501; GRIP=(--gripper-authority 0.0) ;;
    drawer_opening) SEED=9502; GRIP=(--gripper-authority 0.0) ;;
    object_to_drawer) SEED=9503; GRIP=(--gripper-authority 0.0) ;;
    nut_pouring) SEED=9504; GRIP=(--gripper-authority 0.5 --flip-margin 0.5) ;;
  esac
  "$ISAAC_PYTHON" scripts/eval_multitask_residual_v2.py \
    --task "$TASK" \
    --bc-checkpoint "$BC" \
    --residual-checkpoint "$RL" \
    --episodes 20 \
    --num-envs 1 \
    --video-episodes -1 \
    --video-dir "$ROOT/final_eval/${TASK}_seed${SEED}_videos" \
    --output "$ROOT/final_eval/${TASK}_seed${SEED}_20.json" \
    "${GRIP[@]}" \
    --seed "$SEED" \
    --headless \
    --device cuda:0 \
    2>&1 | tee "eval_multitask_residual_${TASK}_seed${SEED}_20.log"
  status=${PIPESTATUS[0]}
  [[ "$status" -eq 0 ]] || exit 1
done

