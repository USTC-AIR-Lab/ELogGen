from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

from eloggen import cli


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "prepare_resources.py"


class PrepareResourcesTest(unittest.TestCase):
    @mock.patch("eloggen.cli.resources_main", return_value=0)
    def test_cli_installs_openarm_resources(self, resource_main: mock.Mock) -> None:
        result = cli.main(
            [
                "resources",
                "install",
                "openarm",
                "--dataset-root",
                "/datasets",
                "--base-url",
                "https://assets.example/v1",
            ]
        )
        self.assertEqual(result, 0)
        command = resource_main.call_args.args[0]
        self.assertIn("openarm-assets", command)
        self.assertIn("/datasets", command)
        self.assertIn("https://assets.example/v1", command)
        self.assertNotIn("--check", command)

    def test_installs_and_checks_resource_from_base_url(self) -> None:
        payload = b"eloggen-resource-test\n"
        relative = Path("src/eloggen/datasets/taskpacks/demo/source/source.hdf5")
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            remote_file = temp / "remote" / relative
            remote_file.parent.mkdir(parents=True)
            remote_file.write_bytes(payload)
            manifest = temp / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "resources": [
                            {
                                "id": "demo_source",
                                "group": "task-data",
                                "root": "project",
                                "path": relative.as_posix(),
                                "size": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            target = temp / "checkout"
            command = [
                sys.executable,
                str(SCRIPT),
                "--manifest",
                str(manifest),
                "--project-root",
                str(target),
                "--group",
                "task-data",
                "--task-data-base-url",
                (temp / "remote").as_uri(),
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual((target / relative).read_bytes(), payload)
            checked = subprocess.run(
                [*command, "--check"], check=False, capture_output=True, text=True
            )
            self.assertEqual(checked.returncode, 0, checked.stderr)


    def test_direct_url_is_used_when_no_mirror_is_configured(self) -> None:
        payload = b"eloggen-direct-resource-test\\n"
        relative = Path("custom_dataset/objects/robot/test.usda")
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            remote_file = temp / "remote" / relative
            remote_file.parent.mkdir(parents=True)
            remote_file.write_bytes(payload)
            manifest = temp / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "resources": [
                            {
                                "id": "direct_asset",
                                "group": "openarm-assets",
                                "root": "dataset",
                                "path": relative.as_posix(),
                                "size": len(payload),
                                "sha256": hashlib.sha256(payload).hexdigest(),
                                "url": remote_file.as_uri(),
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            target = temp / "checkout"
            command = [
                sys.executable,
                str(SCRIPT),
                "--manifest",
                str(manifest),
                "--dataset-root",
                str(target),
                "--group",
                "openarm-assets",
            ]
            installed = subprocess.run(
                command, check=False, capture_output=True, text=True
            )
            self.assertEqual(installed.returncode, 0, installed.stderr)
            self.assertEqual((target / relative).read_bytes(), payload)


if __name__ == "__main__":
    unittest.main()
