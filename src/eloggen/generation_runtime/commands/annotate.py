#!/usr/bin/env python3
"""Annotate V1 stage boundaries for processed ElogGen generation runtime source demos."""

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from eloggen.generation_runtime.context.actions import supported_actions  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detect and validate stage boundaries for processed source HDF5 demos.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", required=True, help="Input processed source HDF5.")
    parser.add_argument("--task", default="openarm_real_exp_1", help="Task name.")
    parser.add_argument("--action", default="pick", choices=supported_actions(), help="High-level GenieSim-style action.")
    parser.add_argument("--demo-key", default="all", help="demo_0 / demo_1 / all.")
    parser.add_argument("--output-dir", required=True, help="Output directory for JSON files.")
    parser.add_argument("--mode", default="validate", choices=["validate"], help="V1 only supports validate.")
    parser.add_argument("--config", default=None, help="Optional base config JSON path.")
    parser.add_argument("--arm", default="left", choices=["left", "right"], help="Active arm.")
    parser.add_argument("--object-name", default=None, help="Passive object id. Defaults to config object_ref.")
    parser.add_argument("--object-label-zh", default=None, help="Chinese object name for generated instructions.")
    parser.add_argument("--object-label-en", default=None, help="English object name for generated instructions.")
    parser.add_argument("--active-object", default=None, help="Active actor id. Defaults to <arm>_gripper.")
    parser.add_argument("--target-name", default=None, help="Optional target object/location id for place/insert actions.")
    parser.add_argument("--target-label-zh", default=None, help="Chinese target name for generated instructions.")
    parser.add_argument("--target-label-en", default=None, help="English target name for generated instructions.")
    parser.add_argument("--eef-object-dist-threshold", type=float, default=None, help="Approach distance threshold.")
    parser.add_argument("--lift-z-threshold", type=float, default=None, help="Lift height threshold.")
    parser.add_argument("--max-manual-auto-diff", type=int, default=None, help="Max ok diff against manual steps.")
    return parser.parse_args()


def main():
    args = parse_args()
    from eloggen.generation_runtime.context.boundary import annotate_stage_boundaries

    detector_config = {}
    if args.eef_object_dist_threshold is not None:
        detector_config["eef_object_dist_threshold"] = args.eef_object_dist_threshold
    if args.lift_z_threshold is not None:
        detector_config["lift_z_threshold"] = args.lift_z_threshold
    if args.max_manual_auto_diff is not None:
        detector_config["max_manual_auto_diff"] = args.max_manual_auto_diff

    result = annotate_stage_boundaries(
        input_path=args.input,
        task_name=args.task,
        output_dir=args.output_dir,
        demo_key=args.demo_key,
        mode=args.mode,
        action=args.action,
        config_path=args.config,
        arm=args.arm,
        object_name=args.object_name,
        object_label_zh=args.object_label_zh,
        object_label_en=args.object_label_en,
        active_object=args.active_object,
        target_name=args.target_name,
        target_label_zh=args.target_label_zh,
        target_label_en=args.target_label_en,
        detector_config=detector_config or None,
    )
    print(json.dumps(result, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
