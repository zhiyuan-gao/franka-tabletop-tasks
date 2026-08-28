"""Success predicates for prismatic drawers and revolute doors."""

from __future__ import annotations


def opening_joint_reached(env, cabinet_cfg, threshold: float):
    cabinet = env.scene[cabinet_cfg.name]
    return cabinet.data.joint_pos[:, cabinet_cfg.joint_ids].squeeze(-1) >= threshold
