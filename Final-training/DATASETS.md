# Dataset inventory

The demonstration datasets are released separately in the private
[Hugging Face dataset repository](https://huggingface.co/datasets/Zhiyuan17/track-b-six-task-demonstrations).
They are intentionally excluded from the GitHub repository because the
complete release is approximately 11 GB. Repository access currently requires
authorization from the owner.

All episodes use synchronized 20 Hz RGB, proprioception, oracle state, reward,
termination flags, timestamps, and normalized 7D actions with a binary
gripper command.

## `datasets/primary_50`

| Task | Successful episodes | Stored-action PhysX replay |
|---|---:|---:|
| `push_object_to_region` | 50 | 14/50 |
| `push_cube_to_region` | 50 | 50/50 |
| `drawer_opening` | 50 | 50/50 |
| `cabinet_opening` | 50 | 50/50 |
| `object_to_drawer` | 50 | 34/50 |
| `nut_pouring` | 50 | 25/50 |

## `datasets/additional_50_seed20260927`

| Task | Successful episodes | Stored-action PhysX replay |
|---|---:|---:|
| `object_to_drawer` | 50 | 31/50 |
| `nut_pouring` | 50 | 23/50 |

Every saved episode succeeded during its original closed-loop PhysX
collection. Stored-action replay is a stricter open-loop repeatability test;
its failures do not change the collection-time success label. Full reports
are stored as `action_replay_report.json` inside each validated dataset.

The two dataset roots should remain separate so episode indices and manifests
do not collide. A behavior-cloning loader can concatenate their episode lists
for ObjectToDrawer and NutPouring.

After downloading, the recommended local layout is:

```text
<repository-directory>/
├── datasets/
│   ├── primary_50/
│   │   ├── push_object_to_region/
│   │   ├── push_cube_to_region/
│   │   ├── drawer_opening/
│   │   ├── cabinet_opening/
│   │   ├── object_to_drawer/
│   │   └── nut_pouring/
│   └── additional_50_seed20260927/
│       ├── object_to_drawer/
│       └── nut_pouring/
├── scripts/
└── src/
```

The `datasets/` directory is ignored by Git. Episode paths recorded in each
manifest are relative to that dataset directory, so downloading or moving the
repository does not require rewriting episode paths.
