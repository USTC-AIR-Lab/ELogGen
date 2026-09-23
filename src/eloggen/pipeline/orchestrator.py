"""Lightweight ElogGen pipeline orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from eloggen.generation.generation_executor import GenerationExecutor
from eloggen.generation.generation_task_spec import export_generation_config
from eloggen.planning.planner import PlanGenerator
from .replay_annotation import replay_annotation_from_task_graph
from .source_parser import task_graph_from_generation_config
from eloggen.planning.recipe import build_trajectory_recipe
from .io import write_json
from .export import (
    check_lerobot_export_imports,
    export_hdf5_call_plan,
    lerobot_export_call_plan,
)
from .generation import check_generation_imports, generation_call_plan
from .run_dir import PipelineRunDir
from .schema import PipelineConfig
from .validate import validate_call_plan, validate_lerobot_dataset


LIGHTWEIGHT_STAGES = {
    "task_graph",
    "logic",
    "recipe",
    "export_config",
    "generate",
    "export_hdf5",
    "export_lerobot",
    "validate",
}

HEAVY_STAGES = {"collect", "prepare_source", "annotate"}


@dataclass
class PipelineStageResult:
    stage: str
    status: str
    outputs: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "outputs": list(self.outputs),
            "details": dict(self.details),
        }


class PipelineOrchestrator:
    def __init__(self, config: PipelineConfig, run_dir: PipelineRunDir):
        self.config = config
        self.run_dir = run_dir
        self.task_graph = None
        self.plans = []
        self.selected_plan = None
        self.recipe = None

    def run(self, stages: Iterable[str], dry_run: bool = True) -> List[PipelineStageResult]:
        results: List[PipelineStageResult] = []
        self.run_dir.write_config(self.config.to_dict())
        for stage in stages:
            result = self._run_stage(stage, dry_run=dry_run)
            results.append(result)
            self.run_dir.write_status({"status": "running", "completed": [item.to_dict() for item in results]})
        self.run_dir.write_status({"status": "complete", "completed": [item.to_dict() for item in results]})
        return results

    def _run_stage(self, stage: str, dry_run: bool) -> PipelineStageResult:
        if stage == "task_graph":
            return self._stage_task_graph(dry_run)
        if stage == "logic":
            return self._stage_logic(dry_run)
        if stage == "recipe":
            return self._stage_recipe(dry_run)
        if stage == "export_config":
            return self._stage_export_config(dry_run)
        if stage == "generate":
            return self._stage_generate(dry_run)
        if stage == "export_lerobot":
            return self._stage_export_lerobot(dry_run)
        if stage == "export_hdf5":
            return self._stage_export_hdf5(dry_run)
        if stage == "validate":
            return self._stage_validate(dry_run)
        # Heavy or unimplemented stages — give a structured planned result
        return self._stage_planned(stage, dry_run)

    # ── lightweight stage implementations ──────────────────────────

    def _stage_task_graph(self, dry_run: bool) -> PipelineStageResult:
        output = self.run_dir.stage_path("task_graph", "task_graph.json")
        details = {
            "adapter": "eloggen.pipeline.source_parser.task_graph_from_generation_config",
            "base_config": self.config.task.base_config,
            "source_demo": self.config.task.source_demo,
        }
        if not dry_run:
            graph = task_graph_from_generation_config(
                self.config.task.base_config, source_demo=self.config.task.source_demo
            )
            self.task_graph = graph
            write_json(output, graph.to_dict())
        return PipelineStageResult(
            "task_graph", "dry_run" if dry_run else "completed", [str(output)], details
        )

    def _stage_logic(self, dry_run: bool) -> PipelineStageResult:
        output = self.run_dir.stage_path("logic", "plans.json")
        details = {
            "adapter": "eloggen.planning.planner.PlanGenerator",
            "max_plans": self.config.logic.max_plans,
            "nearest_principle": self.config.logic.nearest_principle,
        }
        if not dry_run:
            graph = self._ensure_task_graph()
            plans = PlanGenerator(graph).enumerate_plans(max_plans=self.config.logic.max_plans)
            self.plans = plans
            self.selected_plan = self._select_plan(plans)
            write_json(
                output,
                {"plans": [plan.to_dict() for plan in plans], "selected_plan": self.selected_plan.id},
            )
        return PipelineStageResult(
            "logic", "dry_run" if dry_run else "completed", [str(output)], details
        )

    def _stage_recipe(self, dry_run: bool) -> PipelineStageResult:
        output = self.run_dir.stage_path("recipe", "recipe.json")
        details = {
            "adapter": "eloggen.planning.recipe.build_trajectory_recipe",
            "boundary_source": self.config.annotation.boundary_source,
        }
        if not dry_run:
            graph = self._ensure_task_graph()
            plan = self._ensure_plan()
            annotation = replay_annotation_from_task_graph(graph)
            recipe = build_trajectory_recipe(graph, plan, annotation=annotation)
            self.recipe = recipe
            write_json(output, recipe.to_dict())
            details["coverage"] = recipe.boundary_coverage_status
        return PipelineStageResult(
            "recipe", "dry_run" if dry_run else "completed", [str(output)], details
        )

    def _stage_export_config(self, dry_run: bool) -> PipelineStageResult:
        output = self.run_dir.stage_path("export_config", "generation_config.json")
        details = {
            "adapter": "eloggen.generation.generation_task_spec.export_generation_config",
            "base_config": self.config.task.base_config,
        }
        if not dry_run:
            graph = self._ensure_task_graph()
            plan = self._ensure_plan()
            config = export_generation_config(
                self.config.task.base_config,
                graph,
                plan,
                source_demo=self.config.task.source_demo,
                processed_source_demo=self.config.task.processed_source_demo,
                replay_annotation_path=self.config.annotation.annotation_file,
            )
            write_json(output, config)
            details["logic_id"] = plan.id
        return PipelineStageResult(
            "export_config", "dry_run" if dry_run else "completed", [str(output)], details
        )

    def _stage_generate(self, dry_run: bool) -> PipelineStageResult:
        details = generation_call_plan(self.config, self.run_dir)
        import_status = check_generation_imports()
        details["import_status"] = import_status.to_dict()
        output = self.run_dir.stage_path("generate", "generation_stats.json")
        if dry_run:
            status = "dry_run" if import_status.ok else "blocked"
            return PipelineStageResult("generate", status, [str(output)], details)
        if not import_status.ok:
            return PipelineStageResult(
                "generate",
                "blocked",
                [str(output)],
                {
                    **details,
                    "reason": "ElogGen generation runtime generation backend is not importable in the current environment.",
                },
            )
        exported_config = self.run_dir.stage_path("export_config", "generation_config.json")
        if not exported_config.exists():
            self._stage_export_config(dry_run=False)
        source_demo = self.config.task.processed_source_demo or self.config.task.source_demo
        if not source_demo or not Path(source_demo).exists():
            return PipelineStageResult(
                "generate",
                "blocked",
                [str(output)],
                {**details, "reason": f"Processed generation HDF5 not found: {source_demo}"},
            )
        output_folder = details["output_folder"]
        stats = GenerationExecutor().execute(
            exported_config,
            task_name=self.config.task.name,
            source_demo=source_demo,
            scene_file=self.config.task.scene_file,
            output_folder=output_folder,
            num_demos=self.config.generation.num_demos,
            seed=self.config.generation.seed,
            difficulty=self.config.generation.difficulty,
            bimanual=self.config.generation.bimanual,
            robot_type=self.config.generation.robot_type,
            grasp_order=self.config.generation.grasp_order or "task_spec",
            print_stage_type=self.config.generation.print_stage_type,
            output_format=self.config.generation.output_format,
            headless=self.config.generation.headless,
            auto_remove_exp=self.config.generation.auto_remove_exp,
            render=self.config.generation.render,
            no_video_save=self.config.generation.no_video_save,
            video_skip=self.config.generation.video_skip,
            render_image_names=self.config.generation.render_image_names,
            pause_subtask=self.config.generation.pause_subtask,
            enable_marker_vis=self.config.generation.enable_marker_vis,
            ds_ratio=self.config.generation.ds_ratio,
            no_partial_tasks=self.config.generation.no_partial_tasks,
            baseline=self.config.generation.baseline,
            subtask_order=self.config.generation.subtask_order,
            video_fps=self.config.generation.video_fps,
            grasp_order_quota=self.config.generation.grasp_order_quota,
            lerobot_options=self.config.export.lerobot.__dict__,
        )
        write_json(output, stats)
        details["stats"] = stats
        return PipelineStageResult("generate", "completed", [str(output)], details)

    def _stage_export_lerobot(self, dry_run: bool) -> PipelineStageResult:
        details = lerobot_export_call_plan(self.config, self.run_dir)
        import_status = check_lerobot_export_imports()
        details["import_status"] = import_status.to_dict()
        output = self.run_dir.stage_path("export_lerobot", "export_stats.json")
        if dry_run:
            status = "dry_run" if import_status.ok else "blocked"
            return PipelineStageResult("export_lerobot", status, [str(output)], details)
        if not import_status.ok:
            return PipelineStageResult(
                "export_lerobot",
                "blocked",
                [str(output)],
                {
                    **details,
                    "reason": "LeRobot export backend is not importable in the current environment.",
                },
            )
        hdf5_input = details.get("generated_hdf5", "")
        if hdf5_input and not Path(hdf5_input).exists():
            return PipelineStageResult(
                "export_lerobot",
                "blocked",
                [str(output)],
                {
                    **details,
                    "reason": f"Generated HDF5 not found: {hdf5_input}. Run 'generate' stage first.",
                },
            )
        return PipelineStageResult(
            "export_lerobot",
            "planned",
            [str(output)],
            {
                **details,
                "reason": (
                    "Import/API adapter detected the backend and input file, but heavy LeRobot "
                    "conversion execution is intentionally not started by the lightweight runner yet."
                ),
            },
        )

    def _stage_export_hdf5(self, dry_run: bool) -> PipelineStageResult:
        details = export_hdf5_call_plan(self.config, self.run_dir)
        output = self.run_dir.stage_path("export_hdf5", "hdf5_merge_stats.json")
        if dry_run:
            return PipelineStageResult("export_hdf5", "dry_run", [str(output)], details)
        return PipelineStageResult(
            "export_hdf5",
            "planned",
            [str(output)],
            {
                **details,
                "reason": (
                    "HDF5 merge adapter is planned. Heavy merge execution is intentionally "
                    "not started by the lightweight runner yet."
                ),
            },
        )

    def _stage_validate(self, dry_run: bool) -> PipelineStageResult:
        details = validate_call_plan(self.config, self.run_dir)
        output = self.run_dir.stage_path("validate", "validation_result.json")

        if dry_run:
            import_status = check_lerobot_export_imports()
            details["import_status"] = import_status.to_dict()
            return PipelineStageResult(
                "validate", "dry_run" if import_status.ok else "blocked", [str(output)], details
            )

        dataset_paths = details.get("dataset_paths", [])
        validated = False
        validation_result = None
        for ds_path in dataset_paths:
            if Path(ds_path).is_dir():
                validation_result = validate_lerobot_dataset(ds_path)
                validated = True
                details["validation_result"] = validation_result.to_dict()
                break

        if not validated:
            return PipelineStageResult(
                "validate",
                "blocked",
                [str(output)],
                {
                    **details,
                    "reason": (
                        "No LeRobot dataset directory found for validation. "
                        "Checked: " + ", ".join(dataset_paths)
                    ),
                },
            )

        write_json(output, details)
        return PipelineStageResult(
            "validate",
            "completed" if validation_result and validation_result.ok else "failed",
            [str(output)],
            details,
        )

    def _stage_planned(self, stage: str, dry_run: bool) -> PipelineStageResult:
        """Return a structured planned/blocked result for unimplemented heavy stages."""
        details: Dict[str, Any] = {
            "dry_run": dry_run,
            "adapter": f"eloggen.pipeline.{stage}",
        }
        if stage in HEAVY_STAGES:
            details["note"] = (
                f"The '{stage}' stage requires OmniGibson/ROS2/CuRobo runtime. "
                "Its adapter is reserved for a separate collection plugin and is not implemented in this lightweight runner."
            )
        else:
            details["note"] = (
                f"Stage '{stage}' is not recognized by this lightweight runner."
            )
        return PipelineStageResult(stage, "planned", [], details)

    def _ensure_task_graph(self):
        if self.task_graph is None:
            self.task_graph = task_graph_from_generation_config(
                self.config.task.base_config,
                source_demo=self.config.task.source_demo,
            )
        return self.task_graph

    def _ensure_plan(self):
        if self.selected_plan is not None:
            return self.selected_plan
        graph = self._ensure_task_graph()
        plans = self.plans or PlanGenerator(graph).enumerate_plans(max_plans=self.config.logic.max_plans)
        self.plans = plans
        self.selected_plan = self._select_plan(plans)
        return self.selected_plan

    def _select_plan(self, plans):
        if not plans:
            raise ValueError("No execution plans were generated")
        if self.config.logic.plan_id:
            for plan in plans:
                if plan.id == self.config.logic.plan_id:
                    return plan
            available = ", ".join(plan.id for plan in plans)
            raise ValueError(f"Unknown pipeline logic.plan_id {self.config.logic.plan_id!r}. Available: {available}")
        return plans[0]


def parse_stages(stages: Optional[str], default: Iterable[str]) -> List[str]:
    if not stages:
        return list(default)
    return [item.strip() for item in stages.split(",") if item.strip()]
