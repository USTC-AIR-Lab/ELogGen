import tempfile
import unittest
from pathlib import Path

from eloggen.pipeline.run_dir import PipelineRunDir
from eloggen.pipeline.schema import PipelineConfig


class PipelineConfigTests(unittest.TestCase):
    def test_load_openarm_real_exp1_pipeline(self):
        config = PipelineConfig.from_file("src/eloggen/datasets/examples/openarm_real_exp_1.json")
        self.assertEqual(config.task.name, "openarm_real_exp_1")
        self.assertEqual(config.task.family, "object_container_storage")
        self.assertTrue(config.logic.nearest_principle["hard_constraint"])
        self.assertIn("task_graph", config.default_stages())

    def test_run_dir_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = PipelineRunDir.create("task_a", run_dir=str(Path(tmp) / "run"))
            self.assertTrue(run_dir.path("recipes").is_dir())
            self.assertTrue(run_dir.write_config({"a": 1}).exists())
            self.assertTrue(run_dir.write_status({"status": "ok"}).exists())


if __name__ == "__main__":
    unittest.main()
