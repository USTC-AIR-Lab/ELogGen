import unittest

from eloggen.planning.planner import PlanGenerator
from eloggen.pipeline.source_parser import task_graph_from_generation_config


class PlanGeneratorTests(unittest.TestCase):
    def test_openarm_real_exp_1_plans(self):
        graph = task_graph_from_generation_config("src/eloggen/datasets/taskpacks/openarm_real_exp_1/generation.json")
        plans = PlanGenerator(graph).enumerate_plans()
        plan_ids = {plan.id for plan in plans}
        self.assertIn("apple_first", plan_ids)
        self.assertIn("lemon_first", plan_ids)
        self.assertTrue(all(all(plan.constraint_results.values()) for plan in plans))


if __name__ == "__main__":
    unittest.main()

