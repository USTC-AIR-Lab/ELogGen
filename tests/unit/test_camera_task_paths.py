from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest

from eloggen.datasets import taskpack_path
from eloggen.generation_runtime.simulation.camera_config import (
    PUBLISHED_OPENARM_TASKS,
    canonical_openarm_task_name,
    openarm_camera_mount_file_candidates,
    openarm_runtime_difficulty,
    runtime_openarm_task_name,
)


EXPECTED_CAMERA_NAMES = {"left_wrist_cam", "right_wrist_cam", "base_cam"}


class CameraTaskPathTests(unittest.TestCase):
    def test_runtime_variants_resolve_to_task_pack_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            previous = Path.cwd()
            try:
                os.chdir(directory)
                for task_name in sorted(PUBLISHED_OPENARM_TASKS):
                    for runtime_name in (task_name, f"{task_name}_D1"):
                        candidates = openarm_camera_mount_file_candidates(runtime_name)
                        self.assertEqual(Path(candidates[0]), taskpack_path(task_name) / "cameras.json")
                        self.assertTrue(Path(candidates[0]).is_absolute())
                        self.assertTrue(Path(candidates[0]).is_file())
                        self.assertEqual(canonical_openarm_task_name(runtime_name), task_name)
            finally:
                os.chdir(previous)

    def test_all_published_tasks_share_one_runtime_name_rule(self):
        for task_name in sorted(PUBLISHED_OPENARM_TASKS):
            self.assertEqual(runtime_openarm_task_name(task_name), f"{task_name}_D0")
            self.assertEqual(runtime_openarm_task_name(f"{task_name}_D1"), f"{task_name}_D1")
            self.assertEqual(canonical_openarm_task_name(f"{task_name}_D2"), task_name)
            self.assertEqual(openarm_runtime_difficulty(task_name), "D0")
            self.assertEqual(openarm_runtime_difficulty(f"{task_name}_D2"), "D2")

    def test_published_camera_files_are_complete_and_current(self):
        configurations = []
        for task_name in sorted(PUBLISHED_OPENARM_TASKS):
            path = taskpack_path(task_name) / "cameras.json"
            configuration = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(set(configuration), EXPECTED_CAMERA_NAMES)
            for camera in configuration.values():
                self.assertGreaterEqual(set(camera), {"link", "pos", "quat"})
            configurations.append(configuration)

        self.assertEqual(configurations[1:], configurations[:-1])
        self.assertEqual(configurations[0]["left_wrist_cam"]["pos"], [0.103378, -0.002909, 0.104081])
        self.assertEqual(configurations[0]["right_wrist_cam"]["pos"], [0.103378, -0.0015, 0.104081])
        self.assertEqual(configurations[0]["base_cam"]["pos"], [0.101168, 0.002825, 0.700924])


if __name__ == "__main__":
    unittest.main()
