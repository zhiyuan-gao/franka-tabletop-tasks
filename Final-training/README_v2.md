# Track B v2 pipeline — BC + residual RL

Drop-in addition. Nothing existing is overwritten: every new file ends in `_v2`,
and `task_pipeline_v2.py` imports dataset discovery, manifest validation and the
Isaac Lab observation helpers from your existing `task_pipeline_common.py`.

Old checkpoints are **not** compatible — the BC observation and output space
changed on purpose. You retrain BC once (a few hours), then run residual RL.

```
scripts/
  task_pipeline_v2.py        core: dataset, policy, runner, replay, agent
  train_task_bc_v2.py        BC training (no Isaac needed)
  eval_task_bc_v2.py         BC closed-loop eval + best-checkpoint selection
  train_task_residual_v2.py  residual TD3+BC
  eval_task_residual_v2.py   residual eval, BC-vs-residual on the same seeds
  smoke_test_v2.py           full offline dry run against a mock env
```

---

## Run it

```bash
cd /raid/home/than/zhiyuan/ICRA2027/franka-tabletop-tasks
cp <this>/scripts/*_v2.py scripts/
```

### 0. Sanity check (2 min, CPU, no Isaac)

```bash
python scripts/smoke_test_v2.py
```

Runs BC → residual → eval end to end against a mock environment with the same
interface as yours. Do this after any edit before spending GPU time.

### 1. BC

```bash
python scripts/train_task_bc_v2.py \
  --task drawer_opening \
  --output-dir outputs/drawer_bcv2_seed42 \
  --epochs 120 --batch-size 96 --num-workers 8 --amp
```

### 2. Pick the BC checkpoint by rollout success, not validation loss

```bash
python scripts/eval_task_bc_v2.py \
  --task drawer_opening \
  --checkpoint-dir outputs/drawer_bcv2_seed42/checkpoints \
  --episodes 30 --num-envs 8 --headless
```

Writes `outputs/drawer_bcv2_seed42/bc_selection.json` and copies the winner to
`bc_selected.pt`. This is the deployment policy. By default every rollout for
every checkpoint is recorded. Pass `--video-episodes 0` only if you deliberately
want a faster checkpoint sweep without videos.

### 3. Residual RL

```bash
python scripts/train_task_residual_v2.py \
  --task drawer_opening \
  --bc-checkpoint outputs/drawer_bcv2_seed42/bc_selected.pt \
  --output-dir outputs/drawer_residualv2_seed42 \
  --num-envs 8 --transitions 300000 --eval-bc-baseline --headless
```

Deterministic evaluation runs every 10k transitions and `residual_best.pt` is
overwritten only when it improves. You can stop the run at any point and the
best policy is already on disk.

### 4. Final number

```bash
python scripts/eval_task_residual_v2.py \
  --task drawer_opening \
  --residual-checkpoint outputs/drawer_residualv2_seed42/checkpoints/residual_best.pt \
  --episodes 100 --compare-bc --video-episodes -1 --headless
```

`--compare-bc` re-seeds and runs BC-only over the same episodes, so the reported
delta is the residual's actual contribution.  `--video-episodes -1` is the
default and saves one video for every BC and residual episode.  Video recording
forces one environment so the videos correspond exactly to the episodes used
in the reported success rates.  Use `--video-episodes 0` only when videos are
intentionally disabled.

---

## What changed, against each failure in your analysis

### BC

| Problem | Fix |
| --- | --- |
| Single frame, no temporal phase | Frame history: 3 frames at stride 3 (~9 control steps), channel-stacked into one encoder pass. |
| Three-layer CNN much weaker than the CALVIN ResNet-18 | ResNet-18 (GroupNorm) + spatial-softmax keypoints off `layer3` + global pool off `layer4`. Runs at 128×128 so it is cheaper than the old net at 224. |
| No action chunking | Predicts an 8-step chunk; rollout uses ACT-style temporal ensembling (exponential weights over overlapping chunk predictions). This is what makes the policy commit to a motion instead of dithering between phases. |
| Arm actions not normalized | Per-dimension robust min/max (0.5/99.5 percentile + 5% margin) from the train split, stored in the checkpoint, applied identically offline and online. |
| Gripper accuracy is misleading | Gripper BCE is weighted by an exponential bump around every open/close switch, and the logs report `gripper_accuracy_transition` separately from the overall figure. |
| Only demonstration states available | Random-shift (DrQ-v2, same shift across the stack) and brightness/contrast jitter. Doesn't create recovery data, but it stops the encoder from memorizing the demo pixels — the residual RL stage is what supplies recovery behaviour. |
| `bc_best.pt` = best validation loss | The trainer saves periodic epoch checkpoints; `eval_task_bc_v2.py --checkpoint-dir` evaluates all of them on identical seeded resets and selects by success rate. |

Also: EMA weights (used by default at eval), cosine LR with warmup, dropout in
the trunk.

### Residual RL

| Problem | Fix |
| --- | --- |
| Reward hacking: return up, success down | Optimized reward is `0.1 × env_reward + 10 × first_success − 0.02 × flip − 0.01 × ‖arm residual‖²`. Partial opening is worth ~2% of a success now, so "open it a bit and stall" is no longer profitable. |
| `0.99^280 ≈ 0.06` — success invisible from the start state | `--discount 0.997` (`0.997^280 ≈ 0.43`) plus 5-step returns. |
| Truncation treated as termination | Time-limit steps are dropped rather than bootstrapped from the auto-reset observation, and `done` is set only for real terminals. This was silently corrupting every episode's value estimate. |
| BC anchoring decays to 0.2 | TD3+BC normalized λ (`α / mean|Q|`, clamped at 10 so it can't drown the anchor while Q≈0), and the BC weight floors at 0.5 instead of decaying away. |
| Uniform replay dilutes successes | Separate protected success buffer; 25% of every batch is drawn from it. Successes are also copied into a self-imitation bank that the actor regresses onto alongside the demonstrations. |
| ~1 update per 4 transitions | `--updates-per-env-step` defaults to `--num-envs`, i.e. 1 update per transition. |
| Fragile gripper flipping | The flip has to clear a margin (`|base + authority·a_g| > 0.25`), its authority is 0 for the first 40k transitions and then ramps over 40k more, and every flip costs reward. The demonstration residual targets are constructed to reproduce the demo gripper exactly under those same rules, so the anchor teaches correct timing rather than "never flip". |
| No deterministic eval, checkpoints every 20k miss the peak | Deterministic eval every 10k transitions with exploration off; `residual_best.pt` tracks the maximum. |

Exploration is temporally correlated (ρ=0.85) with per-env noise scales from
0.25× to 2× the base, and 10% of episodes run with zero residual so the critic
keeps seeing the base policy and the buffer keeps clean BC data.

---

## What I verified

Everything except the Isaac Sim interaction, which I can't run here:

- `smoke_test_v2.py` runs invariant assertions and then BC → residual → eval end to end against a mock env.
- The online `ChunkRunner` resize and frame/proprio history are asserted to be bit-identical to what
  `ChunkStepDataset` produces offline for the same episode. This was the highest
  risk in the whole design and it's checked.
- N-step returns, discount factors, bootstrap indices and terminal flags are asserted against
  hand-computed values for terminated and truncated cases.
- The action normalizer round-trip is asserted.
- Random shift is asserted to apply the same offset to every frame in a stack.
- `demonstration_residual_target` composed with `compose_action_v2` is asserted to reproduce
  the demonstration gripper command on every frame.
- Mixed replay is asserted to honour the protected-success fraction.
- BC checkpoint sweeps and BC-vs-residual comparisons reset Isaac with the same
  explicit evaluation seed.

What I could not check: the Isaac Lab API calls (`load_registered_env_cfg`,
`configure_normalized_action_space`, `termination_manager.get_term("success")`,
the observation dict layout). Those are copied verbatim from your existing
working scripts, so if BC eval starts, the rest will.

---

## Tuning, in the order I'd try it

1. If BC closed-loop is below ~40%, the problem is still BC. Try
   `--chunk 16 --frame-stack 4`, and `--image-size 160`.
2. If the residual barely moves the number, raise `--arm-residual-limit` to 0.25
   and `--alpha` to 2.5.
3. If it improves then degrades, raise `--bc-weight-end` to 0.8 and
   `--success-fraction` to 0.4.
4. If the gripper is the failure mode in the videos, drop
   `--gripper-warmup-transitions` to 10000 and raise `--flip-penalty` to 0.05.
5. If the run is sim-bound rather than update-bound, `--updates-per-env-step 4`
   with `--num-envs 16`.

One honest expectation: 25% → 55% came from residual RL on a weak BC. Most of
the remaining headroom is in the BC stage, which is why step 1 is step 1. If the
chunked BC lands around 60-70% on its own, residual RL on top of it plausibly
reaches the 90% you want. If BC stays near 30%, no amount of residual tuning
will get there and the next lever is more demonstrations or a second camera.
