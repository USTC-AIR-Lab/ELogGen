import unittest

from eloggen.generation_runtime.simulation.randomization.openarm_drawer_storage import (
    difficulty_from_mapping as drawer_difficulty_from_mapping,
)
from eloggen.generation_runtime.simulation.randomization.openarm_fruit_basket_bagging import (
    difficulty_from_mapping as fruit_difficulty_from_mapping,
)


class RandomizationDifficultyTests(unittest.TestCase):
    def test_drawer_base_task_suffix_uses_configured_difficulty(self):
        spec = drawer_difficulty_from_mapping("STORAGE", {"name": "D0"})
        self.assertEqual(spec.name, "D0")

    def test_drawer_explicit_runtime_difficulty_wins(self):
        spec = drawer_difficulty_from_mapping("D2", {"name": "D0"})
        self.assertEqual(spec.name, "D2")

    def test_fruit_base_task_suffix_uses_configured_difficulty(self):
        spec = fruit_difficulty_from_mapping("BAGGING", {"difficulty": "D1"})
        self.assertEqual(spec.name, "D1")

    def test_unknown_without_config_falls_back_to_d0(self):
        self.assertEqual(drawer_difficulty_from_mapping("STORAGE").name, "D0")
        self.assertEqual(fruit_difficulty_from_mapping("BAGGING").name, "D0")


if __name__ == "__main__":
    unittest.main()
