import unittest

from eloggen.pipeline.context import recipe_context_records
from eloggen.planning.planner import PlanGenerator
from eloggen.pipeline.replay_annotation import replay_annotation_from_task_graph
from eloggen.pipeline.source_parser import task_graph_from_generation_config
from eloggen.planning.recipe import build_trajectory_recipe


class ContextCompatibilityTests(unittest.TestCase):
    def test_recipe_context_usesgeneration_runtime_top_level_fields(self):
        graph = task_graph_from_generation_config(
            "src/eloggen/datasets/taskpacks/openarm_real_exp_1/generation.json",
            source_demo="src/eloggen/datasets/taskpacks/openarm_real_exp_1/source/source.hdf5",
        )
        plan = PlanGenerator(graph).get_plan("lemon_first")
        recipe = build_trajectory_recipe(graph, plan, annotation=replay_annotation_from_task_graph(graph))
        rows = recipe_context_records(recipe)
        self.assertGreater(len(rows), 0)
        first = rows[0]
        for key in (
            "task",
            "action",
            "stage_type",
            "segment",
            "generated_frame",
            "segment_frame",
            "active_arm",
            "active_object",
            "grasp_order",
            "phase_execution_order",
        ):
            self.assertIn(key, first)
        self.assertIn("eloggen", first["metadata"])
        self.assertNotIn("eloggen_logic_id", first)
        self.assertEqual(first["grasp_order"], ["object_2", "object_1"])


if __name__ == "__main__":
    unittest.main()
