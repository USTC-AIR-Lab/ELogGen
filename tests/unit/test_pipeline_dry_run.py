import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from eloggen.pipeline.orchestrator import (
    HEAVY_STAGES,
    LIGHTWEIGHT_STAGES,
    PipelineOrchestrator,
    parse_stages,
)
from eloggen.pipeline.run_dir import PipelineRunDir
from eloggen.pipeline.schema import PipelineConfig


class PipelineDryRunTests(unittest.TestCase):
    def test_parse_stages(self):
        self.assertEqual(
            parse_stages("task_graph, recipe", ["logic"]), ["task_graph", "recipe"]
        )
        self.assertEqual(parse_stages(None, ["logic"]), ["logic"])

    def test_lightweight_stages_known(self):
        """Ensure all lightweight stages are recognised by the orchestrator."""
        self.assertIn("task_graph", LIGHTWEIGHT_STAGES)
        self.assertIn("generate", LIGHTWEIGHT_STAGES)
        self.assertIn("export_lerobot", LIGHTWEIGHT_STAGES)
        self.assertIn("export_hdf5", LIGHTWEIGHT_STAGES)
        self.assertIn("validate", LIGHTWEIGHT_STAGES)

    def test_heavy_stages_known(self):
        self.assertIn("collect", HEAVY_STAGES)
        self.assertIn("prepare_source", HEAVY_STAGES)
        self.assertIn("annotate", HEAVY_STAGES)

    def test_dry_run_core_lightweight_pipeline(self):
        config = PipelineConfig.from_file(
            "src/eloggen/datasets/examples/openarm_real_exp_1.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = PipelineRunDir.create(
                config.task.name, run_dir=str(Path(tmp) / "run")
            )
            results = PipelineOrchestrator(config, run_dir).run(
                ["task_graph", "logic", "recipe", "export_config"],
                dry_run=True,
            )
            self.assertEqual([item.status for item in results], ["dry_run"] * 4)
            self.assertTrue(run_dir.path("config_resolved.json").exists())
            self.assertTrue(run_dir.path("status.json").exists())

    def test_execute_core_lightweight_pipeline(self):
        config = PipelineConfig.from_file(
            "src/eloggen/datasets/examples/openarm_real_exp_1.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = PipelineRunDir.create(
                config.task.name, run_dir=str(Path(tmp) / "run")
            )
            results = PipelineOrchestrator(config, run_dir).run(
                ["task_graph", "logic", "recipe", "export_config"],
                dry_run=False,
            )
            self.assertEqual([item.status for item in results], ["completed"] * 4)
            self.assertTrue(run_dir.path("task_graph.json").exists())
            self.assertTrue(run_dir.path("execution_logic", "plans.json").exists())
            self.assertTrue(run_dir.path("recipes", "recipe.json").exists())
            self.assertTrue(
                run_dir.path("generation_configs", "generation_config.json").exists()
            )

    def test_dry_run_generate_stage(self):
        config = PipelineConfig.from_file(
            "src/eloggen/datasets/examples/openarm_real_exp_1.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = PipelineRunDir.create(
                config.task.name, run_dir=str(Path(tmp) / "run")
            )
            results = PipelineOrchestrator(config, run_dir).run(
                ["generate"],
                dry_run=True,
            )
            self.assertEqual(len(results), 1)
            self.assertIn(results[0].status, ("dry_run", "blocked"))

    def test_generate_executes_internal_backend(self):
        config = PipelineConfig.from_file("src/eloggen/datasets/examples/openarm_real_exp_1.json")
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = PipelineRunDir.create(config.task.name, run_dir=str(Path(tmp) / "run"))
            fake_stats = {"num_success": 1, "num_attempts": 1}
            with patch(
                "eloggen.pipeline.orchestrator.check_generation_imports"
            ) as import_check, patch(
                "eloggen.pipeline.orchestrator.GenerationExecutor.execute",
                return_value=fake_stats,
            ) as execute:
                import_check.return_value.ok = True
                import_check.return_value.to_dict.return_value = {"ok": True}
                result = PipelineOrchestrator(config, run_dir).run(
                    ["task_graph", "logic", "export_config", "generate"],
                    dry_run=False,
                )[-1]
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.details["stats"], fake_stats)
            self.assertTrue(run_dir.stage_path("generate", "generation_stats.json").exists())
            execute.assert_called_once()

    def test_dry_run_export_lerobot_stage(self):
        config = PipelineConfig.from_file(
            "src/eloggen/datasets/examples/openarm_real_exp_1.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = PipelineRunDir.create(
                config.task.name, run_dir=str(Path(tmp) / "run")
            )
            results = PipelineOrchestrator(config, run_dir).run(
                ["export_lerobot"],
                dry_run=True,
            )
            self.assertEqual(len(results), 1)
            self.assertIn(results[0].status, ("dry_run", "blocked"))
            self.assertIn("adapter", results[0].details)

    def test_dry_run_validate_stage(self):
        config = PipelineConfig.from_file(
            "src/eloggen/datasets/examples/openarm_real_exp_1.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = PipelineRunDir.create(
                config.task.name, run_dir=str(Path(tmp) / "run")
            )
            results = PipelineOrchestrator(config, run_dir).run(
                ["validate"],
                dry_run=True,
            )
            self.assertEqual(len(results), 1)
            self.assertIn(results[0].status, ("dry_run", "blocked"))
            self.assertIn("dataset_paths", results[0].details)

    def test_dry_run_heavy_stage_returns_planned(self):
        config = PipelineConfig.from_file(
            "src/eloggen/datasets/examples/openarm_real_exp_1.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = PipelineRunDir.create(
                config.task.name, run_dir=str(Path(tmp) / "run")
            )
            results = PipelineOrchestrator(config, run_dir).run(
                ["collect", "prepare_source", "annotate"],
                dry_run=True,
            )
            self.assertEqual([item.status for item in results], ["planned"] * 3)

    def test_full_lightweight_pipeline_dry_run(self):
        """All lightweight stages pass dry-run without error."""
        config = PipelineConfig.from_file(
            "src/eloggen/datasets/examples/openarm_real_exp_1.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = PipelineRunDir.create(
                config.task.name, run_dir=str(Path(tmp) / "run")
            )
            results = PipelineOrchestrator(config, run_dir).run(
                sorted(LIGHTWEIGHT_STAGES),
                dry_run=True,
            )
            self.assertEqual(len(results), len(LIGHTWEIGHT_STAGES))
            for item in results:
                self.assertIn(
                    item.status,
                    ("dry_run", "blocked", "planned"),
                    f"{item.stage}: unexpected status {item.status}",
                )

    def test_full_pipeline_execute_lightweight_subset(self):
        """Non-dry-run of task_graph → export_config (4 core stages) writes all outputs."""
        config = PipelineConfig.from_file(
            "src/eloggen/datasets/examples/openarm_real_exp_1.json"
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = PipelineRunDir.create(
                config.task.name, run_dir=str(Path(tmp) / "run")
            )
            results = PipelineOrchestrator(config, run_dir).run(
                ["task_graph", "logic", "recipe", "export_config"],
                dry_run=False,
            )
            self.assertEqual([item.status for item in results], ["completed"] * 4)
            # Verify all four output files exist
            self.assertTrue(run_dir.path("task_graph.json").exists())
            self.assertTrue(run_dir.path("execution_logic", "plans.json").exists())
            self.assertTrue(run_dir.path("recipes", "recipe.json").exists())
            self.assertTrue(
                run_dir.path("generation_configs", "generation_config.json").exists()
            )


if __name__ == "__main__":
    unittest.main()