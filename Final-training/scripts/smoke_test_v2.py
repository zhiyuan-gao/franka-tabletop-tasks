#!/usr/bin/env python3
"""End-to-end smoke test for the v2 pipeline with no Isaac Sim installed.

It fabricates a tiny dataset in the released format, stubs ``isaaclab``,
``gymnasium`` and ``track_b_envs`` with a mock environment that follows the same
interface as the real one, then runs:

    train_task_bc_v2.py  ->  train_task_residual_v2.py  ->  eval_task_residual_v2.py

Use it to check that a change did not break shapes, buffers or the training loop
before you burn GPU hours on Denver:

    python scripts/smoke_test_v2.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ORACLE_DIM = 34
EPISODE_STEPS = 40


# --------------------------------------------------------------------------- #
# fake dataset
# --------------------------------------------------------------------------- #
def make_dataset(root: Path, task: str, episodes: int = 8, steps: int = EPISODE_STEPS) -> Path:
    dataset_dir = root / "datasets" / "primary_50" / task
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    manifest_episodes = []
    for index in range(episodes):
        rgb = rng.integers(0, 255, size=(steps, 224, 224, 3), dtype=np.uint8)
        proprio = rng.normal(size=(steps, 25)).astype(np.float32)
        oracle = rng.normal(size=(steps, ORACLE_DIM)).astype(np.float32)
        arm = np.clip(rng.normal(scale=0.15, size=(steps, 6)), -1.0, 1.0).astype(np.float32)
        gripper = np.where(np.arange(steps) < steps // 2, -1.0, 1.0).astype(np.float32)
        action = np.concatenate((arm, gripper[:, None]), axis=1).astype(np.float32)
        reward = rng.normal(size=steps).astype(np.float32)
        terminated = np.zeros(steps, dtype=bool)
        terminated[-1] = True
        truncated = np.zeros(steps, dtype=bool)
        success = np.zeros(steps, dtype=bool)
        success[-1] = True
        timestamp = (np.arange(steps, dtype=np.float32) + 1.0) * 0.05
        name = f"episode_{index:04d}.npz"
        np.savez_compressed(
            dataset_dir / name,
            **{
                "observation.rgb": rgb,
                "observation.proprio": proprio,
                "observation.oracle": oracle,
                "action": action,
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "success": success,
                "timestamp_s": timestamp,
            },
        )
        manifest_episodes.append({"file": name, "episode_index": index, "steps": steps})
    (dataset_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "track_b_successful_demonstrations_v1",
                "task_name": task,
                "episodes": manifest_episodes,
            },
            indent=2,
        )
    )
    return dataset_dir


# --------------------------------------------------------------------------- #
# stubs installed into a sitecustomize-style shim
# --------------------------------------------------------------------------- #
STUB_SOURCE = '''
import sys, types
import numpy as np
import torch

ORACLE_DIM = {oracle_dim}
HORIZON = {horizon}


class MockEnvironment:
    """Same interface as the Isaac Lab manager-based env, tiny and fast."""

    def __init__(self, num_envs=4, device="cpu", seed=0):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.generator = torch.Generator(device="cpu").manual_seed(int(seed))
        self.step_count = torch.zeros(self.num_envs, dtype=torch.long)
        self.progress = torch.zeros(self.num_envs)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool)
        self.unwrapped = self
        self.termination_manager = self

    def get_term(self, name):
        assert name == "success"
        return self._success

    def _observations(self):
        rgb = torch.rand((self.num_envs, 3, 224, 224), generator=self.generator).to(self.device)
        rgb = rgb * 0.2 + self.progress.view(-1, 1, 1, 1).to(self.device) * 0.5
        proprio = torch.randn((self.num_envs, 25), generator=self.generator).to(self.device)
        proprio[:, 0] = self.progress.to(self.device)
        oracle = torch.randn((self.num_envs, ORACLE_DIM), generator=self.generator).to(self.device)
        oracle[:, 0] = self.progress.to(self.device)
        return {{"policy": {{"rgb": rgb.clamp(0.0, 1.0), "proprio": proprio}}, "critic": oracle}}

    def reset(self, seed=None):
        if seed is not None:
            self.generator.manual_seed(int(seed))
        self.step_count.zero_()
        self.progress.zero_()
        self._success.zero_()
        return self._observations(), {{}}

    def step(self, action):
        action = torch.from_numpy(np.asarray(action.detach().cpu().numpy()))
        drive = action[:, :6].abs().mean(dim=1) * (action[:, 6] > 0).float()
        self.progress = (self.progress + 0.05 * drive).clamp(0.0, 1.5)
        self.step_count += 1
        self._success = self.progress > 1.0
        terminated = self._success.clone()
        truncated = (self.step_count >= HORIZON) & ~terminated
        reward = 0.5 * self.progress + 10.0 * self._success.float()
        done = terminated | truncated
        if bool(done.any()):
            self.step_count[done] = 0
            self.progress[done] = 0.0
        return (
            self._observations(),
            reward.to(self.device),
            terminated.to(self.device),
            truncated.to(self.device),
            {{}},
        )

    def close(self):
        return None


class _AppLauncher:
    def __init__(self, args):
        self.app = types.SimpleNamespace(close=lambda: None)

    @staticmethod
    def add_app_launcher_args(parser):
        parser.add_argument("--device", default="cpu")
        parser.add_argument("--disable-fabric", action="store_true")
        parser.add_argument("--enable-cameras", action="store_true")
        parser.add_argument("--headless", action="store_true")


isaaclab = types.ModuleType("isaaclab")
isaaclab_app = types.ModuleType("isaaclab.app")
isaaclab_app.AppLauncher = _AppLauncher
isaaclab.app = isaaclab_app
sys.modules["isaaclab"] = isaaclab
sys.modules["isaaclab.app"] = isaaclab_app

gymnasium = types.ModuleType("gymnasium")
gymnasium.make = lambda task_id, cfg=None: MockEnvironment(**cfg)
sys.modules["gymnasium"] = gymnasium

track_b_envs = types.ModuleType("track_b_envs")
track_b_rl = types.ModuleType("track_b_envs.rl")
track_b_rl.configure_normalized_action_space = lambda env: None
track_b_rl.load_registered_env_cfg = lambda task, device="cpu", num_envs=4, seed=0, use_fabric=True: (
    task,
    {{"num_envs": num_envs, "device": device, "seed": seed}},
)
track_b_envs.rl = track_b_rl
sys.modules["track_b_envs"] = track_b_envs
sys.modules["track_b_envs.rl"] = track_b_rl
'''


INVARIANT_SOURCE = r'''
from pathlib import Path

import numpy as np
import torch

from task_pipeline_common import load_episode_refs
from task_pipeline_v2 import (
    ACTION_DIM,
    ARM_DIM,
    PROPRIO_DIM,
    ActionNormalizer,
    ChunkRunner,
    ChunkStepDataset,
    MixedReplay,
    build_nstep_transitions,
    compose_action_v2,
    demonstration_residual_target,
    random_shift,
)


dataset_dir = Path("datasets/primary_50/drawer_opening")
refs = load_episode_refs([dataset_dir], "drawer_opening")
normalizer = ActionNormalizer.from_refs(refs)

# Action-normalizer round trip.
arm = torch.as_tensor(
    normalizer.low + 0.37 * normalizer.span, dtype=torch.float32
).view(1, ARM_DIM)
round_trip = normalizer.denormalize_torch(normalizer.normalize_torch(arm))
assert torch.allclose(round_trip, arm, atol=1.0e-6), (round_trip, arm)

# Online frame/proprio history must exactly equal the offline dataset view.
offline = ChunkStepDataset(
    refs,
    normalizer,
    frame_stack=2,
    frame_stride=2,
    chunk=4,
    image_size=64,
    gripper_transition_weight_scale=0.0,
    cache_size=2,
)


class CapturePolicy:
    def __init__(self):
        self.frames = None
        self.proprio = None

    def __call__(self, frames, proprio):
        self.frames = frames.detach().clone()
        self.proprio = proprio.detach().clone()
        batch = frames.shape[0]
        return (
            torch.zeros(batch, 4, ARM_DIM),
            torch.zeros(batch, 4),
            torch.zeros(batch, 8),
        )


capture = CapturePolicy()
runner = ChunkRunner(
    capture,
    normalizer,
    num_envs=1,
    device="cpu",
    frame_stack=2,
    frame_stride=2,
    chunk=4,
    image_size=64,
    use_ensemble=False,
)
with np.load(refs[0].file, allow_pickle=False) as episode:
    rgb = torch.from_numpy(episode["observation.rgb"]).permute(0, 3, 1, 2).float() / 255.0
    proprio = torch.from_numpy(episode["observation.proprio"].copy())
for step in range(rgb.shape[0]):
    runner.step(rgb[step : step + 1], proprio[step : step + 1])
    expected = offline[step]
    assert torch.equal(capture.frames[0], expected["frames"].float() / 255.0), step
    assert torch.equal(capture.proprio[0], expected["proprio"]), step

# Random shift must apply the same crop to every frame in a stack.
single = torch.arange(2 * 3 * 16 * 16, dtype=torch.float32).view(2, 1, 3, 16, 16)
stack = single.repeat(1, 3, 1, 1, 1)
torch.manual_seed(7)
shifted = random_shift(stack, pad=3)
assert torch.equal(shifted[:, 0], shifted[:, 1])
assert torch.equal(shifted[:, 1], shifted[:, 2])

# N-step returns, bootstrap indices and terminal semantics.
steps = 3
gamma = 0.9
episode = {
    "feature": np.arange(steps, dtype=np.float32)[:, None],
    "proprio": np.zeros((steps, PROPRIO_DIM), dtype=np.float32),
    "oracle": np.zeros((steps, 2), dtype=np.float32),
    "base_action": np.zeros((steps, ACTION_DIM), dtype=np.float32),
    "residual": np.zeros((steps, ACTION_DIM), dtype=np.float32),
    "reward": np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    "next_feature": np.asarray([[10.0], [20.0], [30.0]], dtype=np.float32),
    "next_proprio": np.zeros((steps, PROPRIO_DIM), dtype=np.float32),
    "next_oracle": np.zeros((steps, 2), dtype=np.float32),
    "next_base_action": np.zeros((steps, ACTION_DIM), dtype=np.float32),
}
terminal = build_nstep_transitions(episode, n_step=2, discount=gamma, terminated_at_end=True)
assert np.allclose(terminal["reward"].ravel(), [2.8, 4.7, 3.0])
assert np.allclose(terminal["discount"].ravel(), [gamma**2, gamma**2, gamma])
assert np.array_equal(terminal["done"].ravel(), [0.0, 1.0, 1.0])
assert np.array_equal(terminal["next_feature"].ravel(), [20.0, 30.0, 30.0])
truncated = build_nstep_transitions(episode, n_step=2, discount=gamma, terminated_at_end=False)
assert np.array_equal(truncated["done"].ravel(), [0.0, 0.0, 0.0])

# Demonstration gripper targets must reproduce the requested binary command.
base = np.zeros((4, ACTION_DIM), dtype=np.float32)
base[:, ARM_DIM] = np.asarray([-1.0, 1.0, -1.0, 1.0])
demo = base.copy()
demo[:, ARM_DIM] = np.asarray([1.0, -1.0, -1.0, 1.0])
target = demonstration_residual_target(demo, base, 0.15, 2.0, 0.25)
composed = compose_action_v2(
    torch.from_numpy(base), torch.from_numpy(target), 2.0, 0.25
)
assert np.array_equal(composed[:, ARM_DIM].numpy(), demo[:, ARM_DIM])

# At least the requested protected-success fraction must appear in a sample.
replay = MixedReplay(256, 128, feature_dim=2, oracle_dim=3, seed=9)


def replay_values(count, reward):
    return {
        "feature": np.zeros((count, 2), dtype=np.float32),
        "proprio": np.zeros((count, PROPRIO_DIM), dtype=np.float32),
        "oracle": np.zeros((count, 3), dtype=np.float32),
        "base_action": np.zeros((count, ACTION_DIM), dtype=np.float32),
        "residual": np.zeros((count, ACTION_DIM), dtype=np.float32),
        "reward": np.full((count, 1), reward, dtype=np.float32),
        "discount": np.full((count, 1), 0.9, dtype=np.float32),
        "done": np.zeros((count, 1), dtype=np.float32),
        "next_feature": np.zeros((count, 2), dtype=np.float32),
        "next_proprio": np.zeros((count, PROPRIO_DIM), dtype=np.float32),
        "next_oracle": np.zeros((count, 3), dtype=np.float32),
        "next_base_action": np.zeros((count, ACTION_DIM), dtype=np.float32),
    }


replay.add_batch(successful=False, **replay_values(128, 0.0))
replay.add_batch(successful=True, **replay_values(64, 7.0))
sample = replay.sample(64, success_fraction=0.25, device="cpu")
assert int((sample["reward"].ravel() == 7.0).sum()) >= 16

print("invariant checks passed")
'''


def write_stub(path: Path) -> None:
    path.write_text(
        "import sitecustomize_stub  # noqa: F401\n"
    )


def run(command: list[str], cwd: Path, env: dict) -> None:
    print("\n$ " + " ".join(command), flush=True)
    result = subprocess.run(command, cwd=cwd, env=env, text=True)
    if result.returncode != 0:
        raise SystemExit(f"command failed with code {result.returncode}")


def main() -> None:
    import os

    task = "drawer_opening"
    root = Path(tempfile.mkdtemp(prefix="trackb_smoke_"))
    print(f"workspace: {root}", flush=True)
    scripts = root / "scripts"
    scripts.mkdir()
    for name in (
        "task_pipeline_common.py",
        "task_pipeline_v2.py",
        "train_task_bc_v2.py",
        "train_task_residual_v2.py",
        "eval_task_residual_v2.py",
        "eval_task_bc_v2.py",
    ):
        shutil.copyfile(HERE / name, scripts / name)
    (scripts / "sitecustomize_stub.py").write_text(
        STUB_SOURCE.format(oracle_dim=ORACLE_DIM, horizon=EPISODE_STEPS)
    )
    (scripts / "usercustomize.py").write_text("import sitecustomize_stub  # noqa: F401\n")
    (scripts / "verify_invariants.py").write_text(INVARIANT_SOURCE)
    make_dataset(root, task, episodes=8, steps=EPISODE_STEPS)

    env = dict(os.environ)
    env["PYTHONPATH"] = str(scripts)
    env["PYTHONSTARTUP"] = ""

    run([sys.executable, "scripts/verify_invariants.py"], root, env)

    bc_out = root / "outputs" / "bc"
    run(
        [
            sys.executable, "-c",
            "import sitecustomize_stub, runpy, sys; "
            "sys.argv[0]='scripts/train_task_bc_v2.py'; "
            "runpy.run_path('scripts/train_task_bc_v2.py', run_name='__main__')",
            "--task", task, "--output-dir", str(bc_out),
            "--epochs", "2", "--batch-size", "8", "--num-workers", "0",
            "--frame-stack", "2", "--frame-stride", "2", "--chunk", "4",
            "--image-size", "64", "--image-feature-dim", "64",
            "--proprio-feature-dim", "32", "--hidden-dim", "64", "--keypoints", "8",
            "--checkpoint-every", "1", "--device", "cpu",
        ],
        root,
        env,
    )

    residual_out = root / "outputs" / "residual"
    run(
        [
            sys.executable, "-c",
            "import sitecustomize_stub, runpy, sys; "
            "sys.argv[0]='scripts/train_task_residual_v2.py'; "
            "runpy.run_path('scripts/train_task_residual_v2.py', run_name='__main__')",
            "--task", task,
            "--bc-checkpoint", str(bc_out / "bc_last.pt"),
            "--output-dir", str(residual_out),
            "--transitions", "3000", "--num-envs", "4",
            "--buffer-size", "5000", "--success-buffer-size", "2000",
            "--batch-size", "32", "--demo-batch-size", "32",
            "--start-training-after", "300", "--random-warmup-transitions", "200",
            "--updates-per-env-step", "1", "--n-step", "3",
            "--eval-interval", "1500", "--eval-episodes", "4",
            "--checkpoint-interval", "1500", "--log-interval", "500",
            "--gripper-warmup-transitions", "500", "--gripper-ramp-transitions", "500",
            "--eval-bc-baseline", "--device", "cpu",
        ],
        root,
        env,
    )

    run(
        [
            sys.executable, "-c",
            "import sitecustomize_stub, runpy, sys; "
            "sys.argv[0]='scripts/eval_task_residual_v2.py'; "
            "runpy.run_path('scripts/eval_task_residual_v2.py', run_name='__main__')",
            "--task", task,
            "--residual-checkpoint", str(residual_out / "checkpoints" / "residual_final.pt"),
            "--episodes", "4", "--num-envs", "4", "--video-episodes", "0",
            "--compare-bc", "--device", "cpu",
        ],
        root,
        env,
    )

    print("\nsmoke test passed", flush=True)
    print(f"artifacts under {root}", flush=True)


if __name__ == "__main__":
    main()
