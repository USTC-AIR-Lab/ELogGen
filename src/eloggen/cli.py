"""Command-line entry point for ElogGen."""

from __future__ import annotations

import argparse
import json
import os
import sys

from eloggen.datasets.resources import main as resources_main
from eloggen.teleoperation import available_collectors, load_collector
from eloggen.pipeline.orchestrator import PipelineOrchestrator, parse_stages
from eloggen.pipeline.run_dir import PipelineRunDir
from eloggen.pipeline.schema import PipelineConfig
from eloggen.pipeline.task_pack import load_task_pack, validate_task_pack


def _forward(module_main, program: str, arguments: list[str]) -> int:
    previous = sys.argv
    try:
        sys.argv = [program, *arguments]
        module_main()
    finally:
        sys.argv = previous
    return 0


def _check_task(name: str) -> int:
    task = load_task_pack(name)
    errors = validate_task_pack(task)
    payload = {
        "task": task.name,
        "root": str(task.root),
        "status": "ok" if not errors else "failed",
        "errors": errors,
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if not errors else 1


def _scene_model_from_task(task) -> str:
    scene_path = task.path("scene")
    scene = json.loads(scene_path.read_text(encoding="utf-8"))
    scene_model = scene.get("init_info", {}).get("args", {}).get("scene_model")
    if not scene_model:
        raise ValueError(f"Task scene does not declare init_info.args.scene_model: {scene_path}")
    return str(scene_model)


def _add_generation_arguments(parser: argparse.ArgumentParser, *, order_alias: bool = False) -> None:
    parser.add_argument("--folder", help="generation output folder override")
    parser.add_argument("--num-demos", "--num_demos", dest="num_demos", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--difficulty", choices=("D0", "D1", "D2"))
    parser.add_argument("--bimanual", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--render", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--no-video-save", "--no_video_save", dest="no_video_save", action="store_true", default=None)
    parser.add_argument("--auto-remove-exp", action="store_true", default=None)
    parser.add_argument("--video-skip", type=int)
    parser.add_argument("--video-fps", type=int)
    parser.add_argument("--render-image-names", nargs="+", default=None)
    parser.add_argument("--pause-subtask", action="store_true", default=None)
    parser.add_argument("--enable-marker-vis", action="store_true", default=None)
    parser.add_argument("--ds-ratio", type=int)
    parser.add_argument("--no-partial-tasks", action="store_true", default=None)
    parser.add_argument("--baseline")
    parser.add_argument("--robot-type", dest="robot_type")
    parser.add_argument("--print-stage-type", action=argparse.BooleanOptionalAction, default=None)
    grasp_flags = ["--grasp-order", "--grasp_order"]
    if order_alias:
        grasp_flags.insert(0, "--order")
    parser.add_argument(*grasp_flags, dest="grasp_order")
    parser.add_argument("--grasp-order-quota", "--grasp_order_quota", dest="grasp_order_quota")
    parser.add_argument("--subtask-order", "--subtask_order", dest="subtask_order")
    parser.add_argument("--output-format", choices=("hdf5", "lerobot", "both"))
    parser.add_argument("--lerobot-output", dest="lerobot_output")
    parser.add_argument("--lerobot-repo-id", dest="lerobot_repo_id")
    parser.add_argument("--lerobot-task", dest="lerobot_task")
    parser.add_argument("--lerobot-robot-type", dest="lerobot_robot_type")
    parser.add_argument("--lerobot-fps", dest="lerobot_fps", type=int)
    parser.add_argument("--lerobot-ordered-tasks", action="store_true", default=None)
    parser.add_argument("--lerobot-overwrite", action="store_true", default=None)
    parser.add_argument("--lerobot-resume", action="store_true", default=None)
    parser.add_argument("--lerobot-checkpoint-every", dest="lerobot_checkpoint_every", type=int)


def _apply_pipeline_overrides(config: PipelineConfig, args: argparse.Namespace) -> None:
    if getattr(args, "task_name", None) is not None:
        config.task.name = args.task_name
    if getattr(args, "source", None) is not None:
        config.task.processed_source_demo = args.source
    if getattr(args, "folder", None) is not None:
        config.generation.folder = args.folder
    if getattr(args, "scene_file", None) is not None:
        config.task.scene_file = args.scene_file

    generation_fields = (
        "num_demos", "seed", "difficulty", "bimanual", "robot_type", "output_format",
        "grasp_order", "grasp_order_quota", "subtask_order", "baseline",
        "print_stage_type", "headless", "auto_remove_exp", "render",
        "no_video_save", "video_skip", "video_fps", "render_image_names",
        "pause_subtask", "enable_marker_vis", "ds_ratio", "no_partial_tasks",
    )
    for field_name in generation_fields:
        value = getattr(args, field_name, None)
        if value is not None:
            setattr(config.generation, field_name, value)

    lerobot_fields = (
        "output", "repo_id", "task", "robot_type", "fps", "ordered_tasks",
        "overwrite", "resume", "checkpoint_every",
    )
    for field_name in lerobot_fields:
        value = getattr(args, "lerobot_" + field_name, None)
        if value is not None:
            setattr(config.export.lerobot, field_name, value)


def _run_pipeline(config: PipelineConfig, stages: str | None, run_dir: str | None, dry_run: bool) -> int:
    resolved_run_dir = PipelineRunDir.create(config.task.name, run_dir=run_dir)
    results = PipelineOrchestrator(config, resolved_run_dir).run(
        parse_stages(stages, config.default_stages()), dry_run=dry_run
    )
    print(json.dumps([item.to_dict() for item in results], indent=2, ensure_ascii=False))
    return 0 if all(item.status not in {"failed", "blocked"} for item in results) else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="eloggen")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check", help="Validate a task pack and source hashes.")
    check.add_argument("task")

    pipeline = subparsers.add_parser("pipeline", help="Run selected ElogGen pipeline stages.")
    pipeline.add_argument("--config", default=None, help="Optional experiment preset. Task-pack mode does not require it.")
    pipeline.add_argument("--task", default=None, help="Build the pipeline directly from this task pack.")
    pipeline.add_argument("--stages", default="task_graph,logic,recipe,export_config")
    pipeline.add_argument("--run-dir", default=None)
    pipeline.add_argument("--dry-run", action="store_true")
    pipeline.add_argument("--task-name", dest="task_name")
    pipeline.add_argument("--source", help="processed source HDF5 override")
    pipeline.add_argument("--scene-file", dest="scene_file")
    _add_generation_arguments(pipeline)

    generate = subparsers.add_parser(
        "generate",
        help="Generate from one task pack without writing a separate pipeline config.",
    )
    generate.add_argument("task")
    generate.add_argument("--run-dir", default=None)
    generate.add_argument("--dry-run", action="store_true")
    generate.add_argument("--plan-id", default=None)
    generate.add_argument("--max-plans", type=int, default=None)
    _add_generation_arguments(generate, order_alias=True)

    scene = subparsers.add_parser("scene", help="Open or smoke-test a task scene.")
    scene.add_argument("task")
    scene.add_argument("--steps", type=int, default=1_000_000)
    scene.add_argument("--headless", action="store_true")

    camera = subparsers.add_parser("camera", help="Tune a task's viewer and training cameras.")
    camera.add_argument("task")

    source = subparsers.add_parser(
        "source",
        help="Prepare a task source HDF5 or inspect replay boundaries.",
    )
    source.add_argument("action", choices=("prepare", "inspect-boundaries"))
    source.add_argument("task")
    source.add_argument("--steps", type=int, default=50)
    source.add_argument("--n", type=int, default=None)
    source.add_argument("--filter-key", default=None)
    source.add_argument("--overwrite", action="store_true")

    assets = subparsers.add_parser("assets", help="Inspect BEHAVIOR-1K assets.")

    resources = subparsers.add_parser(
        "resources", help="Install or verify optional runtime resources."
    )
    resources.add_argument("action", choices=("install", "check"))
    resources.add_argument("resource", choices=("openarm",))
    resources.add_argument("--dataset-root")
    resources.add_argument("--source-root")
    resources.add_argument("--base-url")

    collect = subparsers.add_parser("collect", help="Run an optional collection plugin.")
    collect.add_argument("--list", action="store_true")
    collect.add_argument("--backend")

    args, extra = parser.parse_known_args(argv)

    if args.command == "check":
        return _check_task(args.task)

    if args.command == "pipeline":
        if args.config and args.task:
            parser.error("pipeline accepts either --task or --config, not both")
        if args.config:
            config = PipelineConfig.from_file(args.config)
        else:
            config = PipelineConfig.from_task_pack(args.task or "openarm_real_exp_1")
        _apply_pipeline_overrides(config, args)
        return _run_pipeline(config, args.stages, args.run_dir, args.dry_run)

    if args.command == "generate":
        config = PipelineConfig.from_task_pack(args.task)
        _apply_pipeline_overrides(config, args)
        if args.plan_id is not None:
            config.logic.plan_id = args.plan_id
        if args.max_plans is not None:
            config.logic.max_plans = args.max_plans
        return _run_pipeline(
            config,
            "task_graph,logic,recipe,export_config,generate",
            args.run_dir,
            args.dry_run,
        )

    if args.command == "scene":
        if args.headless:
            os.environ["OMNIGIBSON_HEADLESS"] = "1"
        task = load_task_pack(args.task)
        from eloggen.simulation.scene_viewer import main as scene_main
        return _forward(
            scene_main,
            "eloggen scene",
            ["--template", str(task.path("scene")), "--steps", str(args.steps), *extra],
        )

    if args.command == "camera":
        task = load_task_pack(args.task)
        from eloggen.simulation.camera_tuner import main as camera_main
        return _forward(
            camera_main,
            "eloggen camera",
            [
                "--preset",
                "custom",
                "--task-name",
                task.name,
                "--scene-file",
                str(task.path("scene")),
                "--scene-model",
                _scene_model_from_task(task),
                "--mounts",
                str(task.path("cameras")),
                *extra,
            ],
        )

    if args.command == "source":
        task = load_task_pack(args.task)
        source_path = task.path("source")
        processed_path = task.path("processed_source")
        if args.action == "prepare" and processed_path.exists() and not args.overwrite:
            parser.error(f"Processed source already exists: {processed_path}; pass --overwrite to replace it")

        processing = task.processing
        interface_type = processing.get("interface_type")
        interface_name = processing.get("interface")
        if not interface_type:
            parser.error(f"Task {task.name!r} must declare processing.interface_type in task.yaml")
        if not interface_name and interface_type == "omnigibson_bimanual":
            interface_name = "EG_TaskPackOmniGibsonInterface"
        if not interface_name:
            parser.error(f"Task {task.name!r} must declare processing.interface in task.yaml")

        from eloggen.simulation.omnigibson import activate

        activate()
        from eloggen.generation_runtime.simulation.taskpack_runtime import ensure_task_runtime_registered
        ensure_task_runtime_registered(task.name)
        from eloggen.generation_runtime.commands.prepare import prepare_source_dataset

        prepare_source_dataset(
            dataset_path=str(source_path),
            output_path=str(processed_path),
            env_interface_name=interface_name,
            env_interface_type=interface_type,
            filter_key=args.filter_key,
            n=args.n,
            generate_processed_hdf5=args.action == "prepare",
            replay_for_annotation=args.action == "inspect-boundaries",
            steps=args.steps,
        )
        return 0

    if args.command == "assets":
        from eloggen.simulation.asset_inspector import main as assets_main
        return _forward(assets_main, "eloggen assets", extra)

    if args.command == "resources":
        if extra:
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
        command = ["--group", "openarm-assets"]
        if args.dataset_root:
            command.extend(("--dataset-root", args.dataset_root))
        if args.source_root:
            command.extend(("--source-root", args.source_root))
        if args.base_url:
            command.extend(("--openarm-assets-base-url", args.base_url))
        if args.action == "check":
            command.append("--check")
        return resources_main(command)

    if args.command == "collect":
        if args.list or not args.backend:
            names = sorted(available_collectors())
            print("\n".join(names) if names else "No collection plugins installed.")
            return 0
        load_collector(args.backend)
        print(f"Collection backend {args.backend!r} is installed.")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
