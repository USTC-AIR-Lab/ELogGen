import unittest

from eloggen.pipeline.task_pack import load_task_pack, validate_task_pack


class TaskPackTests(unittest.TestCase):
    def test_published_task_packs_are_complete(self):
        for name in (
            "openarm_real_exp_1",
            "openarm_fruit_basket_bagging",
            "openarm_drawer_storage",
        ):
            with self.subTest(task=name):
                task = load_task_pack(name)
                self.assertEqual(validate_task_pack(task), [])
                self.assertEqual(task.spec["collection"]["interface"], "eloggen.collectors.v1")


if __name__ == "__main__":
    unittest.main()
