import unittest

from eloggen.teleoperation import available_collectors, load_collector


class CollectionInterfaceTests(unittest.TestCase):
    def test_registry_is_available_without_hardware_code(self):
        self.assertIsInstance(available_collectors(), dict)

    def test_missing_backend_has_actionable_error(self):
        with self.assertRaisesRegex(LookupError, "not installed"):
            load_collector("openarm-pico-ros")


if __name__ == "__main__":
    unittest.main()
