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
LATEST="$ROOT/checkpoints/shared_latest.pt"
mkdir -p "$ROOT/checkpoints" "$ROOT/phases"

for ROUND in 1 2 3 4 5 6; do
  for TASK in push_cube_to_region drawer_opening object_to_drawer nut_pouring; do
    PHASE="$ROOT/phases/round_${ROUND}_${TASK}"
    FINAL="$PHASE/checkpoints/residual_final.pt"
    LOG="train_multitask_residual_round_${ROUND}_${TASK}.log"

    if [[ -s "$FINAL" ]]; then
      echo "Already completed: round $ROUND task $TASK"
      cp -f "$FINAL" "$LATEST"
      continue
    fi

    case "$TASK" in
      push_cube_to_region)
        DATA=(--dataset-dir "$PWD/datasets/primary_50/push_cube_to_region")
        GRIP=(--gripper-authority 0.0)
        ;;
      drawer_opening)
        DATA=(--dataset-dir "$PWD/datasets/primary_50/drawer_opening")
        GRIP=(--gripper-authority 0.0)
        ;;
      object_to_drawer)
        DATA=(
          --dataset-dir "$PWD/datasets/primary_50/object_to_drawer"
          --dataset-dir "$PWD/datasets/additional_50_seed20260927/object_to_drawer"
        )
        GRIP=(--gripper-authority 0.0)
        ;;
      nut_pouring)
        DATA=(
          --dataset-dir "$PWD/datasets/primary_50/nut_pouring"
          --dataset-dir "$PWD/datasets/additional_50_seed20260927/nut_pouring"
        )
        GRIP=(
          --gripper-authority 0.5
          --gripper-warmup-transitions 2000
          --gripper-ramp-transitions 8000
          --flip-margin 0.5
          --flip-penalty 0.05
        )
        ;;
    esac

    RESUME=()
    if [[ -s "$LATEST" ]]; then
      RESUME=(--resume-residual-checkpoint "$LATEST")
    fi

    echo "Training shared actor: round $ROUND task $TASK"
    "$ISAAC_PYTHON" scripts/train_multitask_residual_phase_v2.py \
      --task "$TASK" \
      --bc-checkpoint "$BC" \
      "${RESUME[@]}" \
      "${DATA[@]}" \
      "${GRIP[@]}" \
      --output-dir "$PHASE" \
      --transitions 25000 \
      --num-envs 8 \
      --buffer-size 100000 \
      --success-buffer-size 50000 \
      --batch-size 256 \
      --demo-batch-size 256 \
      --start-training-after 2000 \
      --random-warmup-transitions 1000 \
      --updates-per-env-step 2 \
      --arm-residual-limit 0.05 \
      --alpha 0.05 \
      --bc-weight-start 10.0 \
      --bc-weight-end 10.0 \
      --exploration-noise 0.05 \
      --anchor-episode-probability 0.5 \
      --eval-interval 25000 \
      --eval-episodes 10 \
      --eval-seed 4242 \
      --checkpoint-interval 25000 \
      --seed 42 \
      --skip-full-validation \
      --headless \
      --device cuda:0 \
      2>&1 | tee "$LOG"

    status=${PIPESTATUS[0]}
    if [[ "$status" -ne 0 || ! -s "$FINAL" ]]; then
      echo "FAILED: round $ROUND task $TASK"
      exit 1
    fi
    cp -f "$FINAL" "$LATEST"
  done
  cp -f "$LATEST" "$ROOT/checkpoints/shared_round_$(printf '%02d' "$ROUND").pt"
  echo "Completed shared multitask round $ROUND"
done

cp -f "$LATEST" "$ROOT/checkpoints/shared_final.pt"
echo "Training complete: $ROOT/checkpoints/shared_final.pt"

