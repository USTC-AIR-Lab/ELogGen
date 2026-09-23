import unittest

from eloggen.pipeline.export import (
    check_lerobot_export_imports,
    export_hdf5_call_plan,
    lerobot_export_call_plan,
)
from eloggen.pipeline.run_dir import PipelineRunDir
from eloggen.pipeline.schema import PipelineConfig


class LeRobotExportAdapterTests(unittest.TestCase):
    def test_lerobot_export_import_check_is_structured(self):
        status = check_lerobot_export_imports()
        data = status.to_dict()
        self.assertIn("ok", data)
        self.assertIn("module", data)
        if not status.ok:
            self.assertIsNotNone(status.error_type)

    def test_lerobot_export_call_plan(self):
        config = PipelineConfig.from_file(
            "src/eloggen/datasets/examples/openarm_real_exp_1.json"
        )
        run_dir = PipelineRunDir.create(
            config.task.name, run_dir="/tmp/eloggen_export_adapter_test"
        )
        plan = lerobot_export_call_plan(config, run_dir)
        self.assertEqual(plan["adapter"], "eloggen.pipeline.export")
        self.assertEqual(plan["backend"], "eloggen.generation_runtime.commands.export")
        self.assertEqual(plan["robot_type"], "openarm")
        self.assertEqual(plan["ordered_tasks"], True)
        self.assertIn("output", plan)

    def test_export_hdf5_call_plan(self):
        config = PipelineConfig.from_file(
            "src/eloggen/datasets/examples/openarm_real_exp_1.json"
        )
        run_dir = PipelineRunDir.create(
            config.task.name, run_dir="/tmp/eloggen_hdf5_export_test"
        )
        plan = export_hdf5_call_plan(config, run_dir)
        self.assertEqual(plan["adapter"], "eloggen.pipeline.export")
        self.assertEqual(plan["backend"], "eloggen.generation_runtime.commands.merge")
        self.assertIn("output", plan)

    def test_lerobot_export_with_enabled_config(self):
        """Validate that an enabled lerobot export config produces a consistent call plan."""
        config = PipelineConfig.from_file(
            "src/eloggen/datasets/examples/openarm_real_exp_1.json"
        )
        # Override to simulate an enabled export
        config.export.lerobot.enabled = True
        config.export.lerobot.output = "/tmp/fake_lerobot_output"
        run_dir = PipelineRunDir.create(
            config.task.name, run_dir="/tmp/eloggen_enabled_export_test"
        )
        plan = lerobot_export_call_plan(config, run_dir)
        self.assertEqual(plan["output"], "/tmp/fake_lerobot_output")
        self.assertEqual(plan["repo_id"], "openarm/openarm_real_exp_1")


if __name__ == "__main__":
    unittest.main()