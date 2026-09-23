import json
import tempfile
import unittest
from pathlib import Path

import h5py

from eloggen.generation.generation_executor import (
    load_runtime_config_payload,
    runtime_environment_name,
    validate_generation_source,
)
from eloggen.datasets import taskpack_path
from eloggen.generation_runtime.interface_names import (
    canonicalize_env_interface_name,
    get_canonical_env_interface_info_from_dataset,
)
from eloggen.pipeline.generation import check_generation_imports, generation_call_plan
from eloggen.pipeline.run_dir import PipelineRunDir
from eloggen.pipeline.schema import PipelineConfig


class GenerationAdapterTests(unittest.TestCase):
    def test_generation_import_check_is_structured(self):
        status = check_generation_imports()
        data = status.to_dict()
        self.assertIn("ok", data)
        self.assertIn("module", data)
        if not status.ok:
            self.assertIsNotNone(status.error_type)

    def test_locked_config_optional_key_uses_membership_check(self):
        from robomimic.config.config import Config

        config = Config({"name": "demo"})
        config.lock_keys()
        self.assertNotIn("subtask_graph", config)
        with self.assertRaisesRegex(RuntimeError, "subtask_graph"):
            _ = config.subtask_graph

    def test_runtime_payload_strips_eloggen_provenance(self):
        payload = {"name": "demo", "type": "omnigibson_bimanual", "eloggen": {"logic_id": "x"}, "meta": {}}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps(payload))
            runtime = load_runtime_config_payload(path)
        self.assertNotIn("eloggen", runtime)
        self.assertNotIn("meta", runtime)
        self.assertEqual(runtime["name"], "demo")

    def test_legacy_mg_interface_names_are_canonicalized_to_eg(self):
        self.assertEqual(
            canonicalize_env_interface_name("MG_OpenArmDrawerStorage"),
            "EG_OpenArmDrawerStorageInterface",
        )
        self.assertEqual(
            canonicalize_env_interface_name("MG_OpenArmFruitBasketBagging"),
            "EG_OpenArmFruitBasketBaggingInterface",
        )
        self.assertEqual(
            canonicalize_env_interface_name(b"MG_OpenArmRealExp1"),
            "EG_OpenArmRealExp1Interface",
        )
        self.assertEqual(
            canonicalize_env_interface_name("EG_OpenArmDrawerStorageInterface"),
            "EG_OpenArmDrawerStorageInterface",
        )

    def test_all_published_base_tasks_default_to_d0_runtime_variant(self):
        for task_name in (
            "openarm_real_exp_1",
            "openarm_drawer_storage",
            "openarm_fruit_basket_bagging",
        ):
            self.assertEqual(runtime_environment_name(task_name), f"{task_name}_D0")
            self.assertEqual(runtime_environment_name(f"{task_name}_D2"), f"{task_name}_D2")

    def test_mixed_legacy_and_canonical_hdf5_names_compare_after_canonicalization(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "processed.hdf5"
            with h5py.File(path, "w") as stream:
                for demo_key, interface_name in (
                    ("demo_0", "MG_OpenArmRealExp1"),
                    ("demo_1", "EG_OpenArmRealExp1Interface"),
                ):
                    group = stream.require_group(f"data/{demo_key}/datagen_info")
                    group.attrs["env_interface_name"] = interface_name
                    group.attrs["env_interface_type"] = "omnigibson_bimanual"

            interface_name, interface_type = get_canonical_env_interface_info_from_dataset(
                path,
                ["demo_0", "demo_1"],
            )

        self.assertEqual(interface_name, "EG_OpenArmRealExp1Interface")
        self.assertEqual(interface_type, "omnigibson_bimanual")

    def test_generation_source_requires_datagen_info(self):
        with self.assertRaisesRegex(ValueError, "processed source HDF5"):
            validate_generation_source("src/eloggen/datasets/taskpacks/openarm_real_exp_1/source/source.hdf5")
        path = validate_generation_source("src/eloggen/datasets/taskpacks/openarm_real_exp_1/source/processed.hdf5")
        self.assertEqual(path.name, "processed.hdf5")

    def test_generation_call_plan(self):
        config = PipelineConfig.from_file("src/eloggen/datasets/examples/openarm_real_exp_1.json")
        run_dir = PipelineRunDir.create(config.task.name, run_dir="/tmp/eloggen_generation_adapter_test")
        plan = generation_call_plan(config, run_dir)
        self.assertEqual(plan["backend"], "eloggen.generation_runtime.commands.generate")
        self.assertEqual(
            plan["source_demo"],
            str(taskpack_path("openarm_real_exp_1") / "source/processed.hdf5"),
        )
        self.assertEqual(
            plan["raw_source_demo"],
            str(taskpack_path("openarm_real_exp_1") / "source/source.hdf5"),
        )
        self.assertEqual(plan["robot_type"], "OpenArm")


if __name__ == "__main__":
    unittest.main()
