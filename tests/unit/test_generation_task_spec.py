import unittest

from eloggen.generation.generation_task_spec import export_generation_config
from eloggen.planning.planner import PlanGenerator
from eloggen.pipeline.source_parser import task_graph_from_generation_config


class GenerationTaskSpecTests(unittest.TestCase):
    def test_export_keeps_plan_metadata(self):
        base = "src/eloggen/datasets/taskpacks/openarm_real_exp_1/generation.json"
        graph = task_graph_from_generation_config(base)
        plan = PlanGenerator(graph).get_plan("apple_first")
        config = export_generation_config(base, graph, plan)
        self.assertEqual(config["eloggen"]["plan_id"], "apple_first")
        self.assertIn("trajectory_recipe", config["eloggen"])
        self.assertIn("phase_1", config["task"]["task_spec"])

    def test_export_preserves_drawer_source_boundary_order(self):
        base = "src/eloggen/datasets/taskpacks/openarm_drawer_storage/generation.json"
        graph = task_graph_from_generation_config(base)
        plan = PlanGenerator(graph).enumerate_plans(max_plans=1)[0]
        config = export_generation_config(base, graph, plan)

        task_spec = config["task"]["task_spec"]
        term_steps = [
            phase["arm_left"]["subtask_1"]["subtask_term_step"]
            for phase in task_spec.values()
        ]
        self.assertEqual(term_steps, [580, 1120, 1300, 1620, 1820])
        self.assertEqual(term_steps, sorted(term_steps))
        self.assertEqual(config["eloggen"]["phase_order"], list(plan.phase_order))


if __name__ == "__main__":
    unittest.main()
