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
RESULTS="$ROOT/selection_seed4242"
mkdir -p "$RESULTS"

for CKPT in "$ROOT"/checkpoints/shared_round_*.pt; do
  STEM="$(basename "$CKPT" .pt)"
  for TASK in push_cube_to_region drawer_opening object_to_drawer nut_pouring; do
    OUT="$RESULTS/${STEM}__${TASK}.json"
    [[ -s "$OUT" ]] && { echo "Already completed: $OUT"; continue; }
    if [[ "$TASK" == nut_pouring ]]; then
      GRIP=(--gripper-authority 0.5 --flip-margin 0.5)
    else
      GRIP=(--gripper-authority 0.0)
    fi
    "$ISAAC_PYTHON" scripts/eval_multitask_residual_v2.py \
      --task "$TASK" \
      --bc-checkpoint "$BC" \
      --residual-checkpoint "$CKPT" \
      --episodes 10 \
      --num-envs 8 \
      --video-episodes 0 \
      "${GRIP[@]}" \
      --seed 4242 \
      --output "$OUT" \
      --headless \
      --device cuda:0 \
      2>&1 | tee -a select_multitask_residual_seed4242.log
    status=${PIPESTATUS[0]}
    [[ "$status" -eq 0 && -s "$OUT" ]] || exit 1
  done
done

python - <<'PY'
import json
import re
import shutil
from pathlib import Path

root = Path("outputs/multitask_residualv2_epoch240_seed42")
results = root / "selection_seed4242"
tasks = ["push_cube_to_region", "drawer_opening", "object_to_drawer", "nut_pouring"]
rows = {}
for path in results.glob("shared_round_*__*.json"):
    match = re.match(r"shared_round_(\d+)__(.+)\.json", path.name)
    if not match:
        continue
    round_id, task = int(match.group(1)), match.group(2)
    data = json.loads(path.read_text())
    row = rows.setdefault(round_id, {"round": round_id, "task_success_rates": {}})
    row["task_success_rates"][task] = float(data["residual"]["success_rate"])

complete = [row for row in rows.values() if all(t in row["task_success_rates"] for t in tasks)]
if not complete:
    raise RuntimeError("no round completed evaluation on all four tasks")
for row in complete:
    row["mean_task_success_rate"] = sum(row["task_success_rates"][t] for t in tasks) / len(tasks)
complete.sort(key=lambda row: (-row["mean_task_success_rate"], row["round"]))
best = complete[0]
source = root / "checkpoints" / f"shared_round_{best['round']:02d}.pt"
target = root / "checkpoints" / "shared_best.pt"
shutil.copyfile(source, target)
summary = {
    "selection_seed": 4242,
    "episodes_per_task": 10,
    "best_round": best["round"],
    "best_mean_task_success_rate": best["mean_task_success_rate"],
    "best_task_success_rates": best["task_success_rates"],
    "selected_checkpoint": str(target.resolve()),
    "all_rounds": complete,
}
(root / "selection_summary_seed4242.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

