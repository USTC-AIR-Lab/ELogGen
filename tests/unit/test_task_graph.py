import unittest

from eloggen.pipeline.source_parser import task_graph_from_generation_config


class TaskGraphTests(unittest.TestCase):
    def test_openarm_real_exp_1_parse(self):
        graph = task_graph_from_generation_config("src/eloggen/datasets/taskpacks/openarm_real_exp_1/generation.json")
        self.assertEqual(graph.name, "openarm_real_exp_1")
        self.assertEqual({b.object for b in graph.bindings}, {"object_1", "object_2"})
        self.assertEqual({b.target for b in graph.bindings}, {"paper_bag_1"})


if __name__ == "__main__":
    unittest.main()

