import unittest


class GenerationRuntimeImportTests(unittest.TestCase):
    def test_context_and_lerobot_exports_are_importable(self):
        from eloggen.simulation.omnigibson import activate
        activate()
        from eloggen.generation_runtime.context.hdf5 import write_frame_context_json
        from eloggen.generation_runtime.commands.export import LeRobotDatasetWriter, validate_lerobot_dataset

        self.assertEqual(write_frame_context_json.__name__, "write_frame_context_json")
        self.assertEqual(LeRobotDatasetWriter.__name__, "LeRobotDatasetWriter")
        self.assertEqual(validate_lerobot_dataset.__name__, "validate_lerobot_dataset")


if __name__ == "__main__":
    unittest.main()
