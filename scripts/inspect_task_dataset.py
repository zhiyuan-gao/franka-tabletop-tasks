#!/usr/bin/env python3
"""Validate a released Track B demonstration dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from task_pipeline_common import (
    TASK_NAMES,
    inspect_datasets,
    json_print,
    resolve_dataset_dirs,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_NAMES, required=True)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        action="append",
        default=None,
        help="Task dataset directory containing manifest.json. Repeat to concatenate roots.",
    )
    args = parser.parse_args()
    args.dataset_dir = resolve_dataset_dirs(args.task, args.dataset_dir)
    json_print(inspect_datasets(args.dataset_dir, args.task))


if __name__ == "__main__":
    main()
