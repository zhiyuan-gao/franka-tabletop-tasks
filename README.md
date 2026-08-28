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
