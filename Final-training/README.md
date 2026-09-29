# Franka Tabletop Tasks

Six standalone Isaac Lab manipulation environments for demonstration
collection, imitation learning, and SKRL reinforcement learning. The package
uses a single RGB policy camera, Franka proprioception, and optional
simulator-only state for the critic.

[View one successful demonstration for each task](https://zhiyuan-gao.github.io/franka-tabletop-tasks/demos/).

## Requirements

Tested on:

- Ubuntu 22.04
- Python 3.11
- Isaac Sim 5.1
- Isaac Lab 2.3.0
- an NVIDIA RTX GPU supported by Isaac Sim

Install and verify Isaac Sim and Isaac Lab first by following the
[Isaac Lab 2.3.0 installation guide](https://isaac-sim.github.io/IsaacLab/v2.3.0/source/setup/installation/index.html).
Run the commands below inside the same Python environment used by Isaac Lab.

## Installation

```bash
git clone git@github.com:zhiyuan-gao/franka-tabletop-tasks.git
cd franka-tabletop-tasks

# Environment package and demonstration tools
python -m pip install -e .

# Also install the supported SKRL training dependencies
python -m pip install -e ".[rl]"
```

The repository contains the project-specific USD, texture, calibration, and
background files required at runtime. The first run still needs internet
access to download standard Isaac assets such as the Franka and YCB sugar box.

## Running on the ICRA2027 workstation

This section records the machine-specific paths used on the current ICRA2027
workstation. It supplements the portable installation instructions above and
is not required on other machines.

The authoritative checkout, downloaded demonstrations, and Isaac Lab Python
executable are located at:

```text
Repository:
/raid/home/than/zhiyuan/ICRA2027/franka-tabletop-tasks

Dataset:
/raid/home/than/zhiyuan/ICRA2027/franka-tabletop-tasks/datasets

Miniforge/Conda environment:
lw_benchhub

Isaac Lab Python:
/raid/home/than/miniforge3/envs/lw_benchhub/bin/python
```

This workstation uses the Miniforge/Conda environment `lw_benchhub`.
`track-b-envs` is installed with pip in editable mode inside that environment.
The repository does not use uv on this workstation. Do not use the system
Python, because it does not provide the matching Isaac Sim and Isaac Lab
packages. Activate the environment and enter the repository with:

```bash
source /raid/home/than/miniforge3/etc/profile.d/conda.sh
conda activate lw_benchhub

export TRACK_B_REPO=/raid/home/than/zhiyuan/ICRA2027/franka-tabletop-tasks
export ISAAC_PYTHON=/raid/home/than/miniforge3/envs/lw_benchhub/bin/python
cd "$TRACK_B_REPO"
```

Verify the editable installation and the six task registrations with:

```bash
"$ISAAC_PYTHON" -c \
  "import track_b_envs; print(track_b_envs.TASK_IDS)"
```

The following CPU command has been validated as a complete, minimal PPO
update on this workstation:

```bash
"$ISAAC_PYTHON" scripts/rl/train.py \
  --task push_cube_to_region \
  --smoke \
  --headless \
  --device cpu
```

Use a CUDA device for a normal training run:

```bash
"$ISAAC_PYTHON" scripts/rl/train.py \
  --task push_object_to_region \
  --num-envs 4 \
  --headless \
  --device cuda:0
```

Training outputs are written under `outputs/rl/skrl/` unless `--log-dir` is
specified. The local `datasets/` and `outputs/` directories are ignored by
Git and must not be committed.

### Local demonstration dataset

The downloaded private Hugging Face dataset occupies approximately 11 GB and
contains 400 successful episodes. Its local layout is:

```text
datasets/
├── primary_50/
│   ├── push_object_to_region/       # 50 episodes
│   ├── push_cube_to_region/         # 50 episodes
│   ├── drawer_opening/              # 50 episodes
│   ├── cabinet_opening/             # 50 episodes
│   ├── object_to_drawer/            # 50 episodes
│   └── nut_pouring/                 # 50 episodes
└── additional_50_seed20260927/
    ├── object_to_drawer/            # 50 additional episodes
    └── nut_pouring/                 # 50 additional episodes
```

Each task directory contains:

```text
manifest.json
action_replay_report.json
episodes/
└── episode_000000.npz
```

Each compressed episode stores synchronized arrays:

| Key | Shape and content |
|---|---|
| `observation.rgb` | `uint8 [T, 224, 224, 3]` composite policy RGB |
| `observation.proprio` | `float32 [T, 25]` deployable Franka state |
| `observation.oracle` | `float32 [T, task-dependent]` simulator-only state |
| `action` | `float32 [T, 7]` normalized relative Cartesian and binary-gripper action |
| `reward` | Per-step environment reward |
| `terminated`, `truncated`, `success` | Per-step episode flags |
| `timestamp_s` | Monotonic 20 Hz control timestamps |

`manifest.json` records task metadata, episode paths, reset seeds, and initial
states. `action_replay_report.json` records the stored-action PhysX replay
audit. The visual actor may consume only `observation.rgb` and
`observation.proprio`; the oracle state is reserved for the critic, experts,
analysis, and debugging.

The dataset repository is private. Never place a Hugging Face access token in
this README, the Git repository, scripts, or committed configuration files.

The validated CPU smoke run can print PhysX warnings when a reset event tries
to write velocity to a kinematic scene object. These warnings did not prevent
the PPO smoke update from completing, but the reset behavior should be audited
before large training runs.

## Verify the installation

First verify that the package imports and all six tasks are registered:

```bash
python -c "import track_b_envs; print(track_b_envs.TASK_IDS)"
```

Then run one short headless training smoke test:

```bash
python scripts/rl/train.py \
  --task push_cube_to_region \
  --smoke \
  --headless
```

To render a task in the Isaac Sim UI, omit `--headless`:

```bash
python scripts/rl/play.py \
  --task push_cube_to_region \
  --checkpoint <checkpoint-path>
```

## Collect one demonstration

```bash
python scripts/collect_successful_demos.py \
  --task push_cube_to_region \
  --output-dir outputs/push_cube_demo \
  --num-episodes 1 \
  --headless
```

Supported task names are `push_object_to_region`, `push_cube_to_region`,
`drawer_opening`, `cabinet_opening`, `object_to_drawer`, and `nut_pouring`.

## Documentation and data

- [Environment and training interface](docs/rl_environments.md)
- [Task definitions](docs/task_definitions.md)
- [Demonstration collection](docs/data_collection.md)
- [Project architecture](docs/architecture.md)
- [Dataset download and layout](DATASETS.md)
- [Demonstration dataset on Hugging Face](https://huggingface.co/datasets/Zhiyuan17/track-b-six-task-demonstrations)

The GitHub repository contains code and runtime assets only. Demonstration
episodes, checkpoints, logs, and generated videos are not committed.
