#!/usr/bin/env python3
"""Validate the released drawer-opening demonstration dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from drawer_pipeline_common import inspect_datasets, json_print


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        action="append",
        required=True,
        help="Drawer dataset directory containing manifest.json. Repeat to concatenate roots.",
    )
    args = parser.parse_args()
    json_print(inspect_datasets(args.dataset_dir))


if __name__ == "__main__":
    main()
