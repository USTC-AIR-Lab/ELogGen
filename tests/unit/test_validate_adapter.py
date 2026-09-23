import tempfile
import unittest
from pathlib import Path

from eloggen.pipeline.validate import validate_call_plan, validate_lerobot_dataset
from eloggen.pipeline.run_dir import PipelineRunDir
from eloggen.pipeline.schema import PipelineConfig


class ValidateAdapterTests(unittest.TestCase):
    def test_validate_call_plan(self):
        config = PipelineConfig.from_file(
            "src/eloggen/datasets/examples/openarm_real_exp_1.json"
        )
        run_dir = PipelineRunDir.create(
            config.task.name, run_dir="/tmp/eloggen_validate_adapter_test"
        )
        plan = validate_call_plan(config, run_dir)
        self.assertEqual(plan["adapter"], "eloggen.pipeline.validate")
        self.assertIn("checks", plan)
        self.assertIn("dataset_paths", plan)
        self.assertGreater(len(plan["checks"]), 0)

    def test_validate_missing_directory(self):
        """Validation of a non-existent path should report failures, not crash."""
        result = validate_lerobot_dataset("/tmp/nonexistent_lerobot_dataset_xyz")
        self.assertFalse(result.ok)
        self.assertGreater(len(result.errors), 0)
        # Should still return structured output
        self.assertIn("meta_dir_exists", [c["name"] for c in result.checks])

    def test_validate_empty_directory(self):
        """Validation of an empty directory should report missing meta/."""
        with tempfile.TemporaryDirectory() as tmp:
            result = validate_lerobot_dataset(tmp)
            self.assertFalse(result.ok)
            self.assertIn("meta_dir_exists", [c["name"] for c in result.checks])
            self.assertEqual(
                result.checks[0]["status"], "failed"
            )  # meta_dir_exists should fail

    def test_validate_with_partial_structure(self):
        """Validation of a directory with meta/ but no data/ should report specific failures."""
        with tempfile.TemporaryDirectory() as tmp:
            meta_dir = Path(tmp) / "meta"
            meta_dir.mkdir()
            (meta_dir / "info.json").write_text('{"total_episodes": 5}')
            result = validate_lerobot_dataset(tmp)
            self.assertFalse(result.ok)
            # meta/ checks should pass (they exist)
            check_names = [c["name"] for c in result.checks]
            self.assertIn("meta_dir_exists", check_names)
            self.assertIn("meta/info.json exists", check_names)
            self.assertIn("info.json total_episodes parsed", check_names)
            # data/ and videos/ dirs don't exist → checks should fail
            self.assertIn("data/ dir exists", check_names)
            self.assertIn("videos/ dir exists", check_names)
            # meta tasks / episodes / semantic_context should also fail
            self.assertIn("meta/tasks.jsonl exists", check_names)
            self.assertIn("meta/episodes.jsonl exists", check_names)

    def test_result_to_dict(self):
        result = validate_lerobot_dataset("/tmp/nonexistent_xyz")
        data = result.to_dict()
        self.assertIsInstance(data, dict)
        self.assertIn("ok", data)
        self.assertIn("checks", data)
        self.assertIn("summary", data)
        self.assertIn("errors", data)


if __name__ == "__main__":
    unittest.main()