import unittest

from eloggen.generation.generation_task_spec import export_generation_config
from eloggen.pipeline.context import recipe_context_records
from eloggen.pipeline.boundaries import extract_boundary_states_from_hdf5, validate_replay_annotation
from eloggen.planning.planner import PlanGenerator
from eloggen.pipeline.replay_annotation import replay_annotation_from_task_graph
from eloggen.pipeline.source_parser import task_graph_from_generation_config
from eloggen.planning.recipe import build_trajectory_recipe


class ReplayRecipeTests(unittest.TestCase):
    def test_manual_boundaries_cover_real_exp1_lemon_first(self):
        graph = task_graph_from_generation_config(
            "src/eloggen/datasets/taskpacks/openarm_real_exp_1/generation.json",
            source_demo="src/eloggen/datasets/taskpacks/openarm_real_exp_1/source/source.hdf5",
        )
        plan = PlanGenerator(graph).get_plan("lemon_first")
        annotation = replay_annotation_from_task_graph(graph)
        errors = validate_replay_annotation(annotation)
        self.assertEqual(errors, [])
        boundary_states = extract_boundary_states_from_hdf5("src/eloggen/datasets/taskpacks/openarm_real_exp_1/source/source.hdf5", annotation)
        recipe = build_trajectory_recipe(graph, plan, annotation=annotation, boundary_states=boundary_states)
        self.assertEqual(recipe.boundary_coverage_status, "covered")
        replay_segments = [segment for segment in recipe.ordered_segments if segment.type == "replay"]
        self.assertEqual(len(replay_segments), 4)
        self.assertEqual(replay_segments[0].source_frame_range, [210, 420])
        context = recipe_context_records(recipe)
        self.assertIn("stage_type", context[0])
        self.assertIn("active_object", context[0])
        self.assertTrue(any(item["metadata"]["eloggen"]["is_replay_frame"] for item in context))
        self.assertTrue(any(item["grasp_order"] == ["object_2", "object_1"] for item in context))

    def test_export_includes_trajectory_recipe(self):
        base = "src/eloggen/datasets/taskpacks/openarm_real_exp_1/generation.json"
        graph = task_graph_from_generation_config(base, source_demo="src/eloggen/datasets/taskpacks/openarm_real_exp_1/source/source.hdf5")
        plan = PlanGenerator(graph).get_plan("lemon_first")
        config = export_generation_config(
            base,
            graph,
            plan,
            source_demo="src/eloggen/datasets/taskpacks/openarm_real_exp_1/source/source.hdf5",
        )
        self.assertIn("trajectory_recipe", config["eloggen"])
        self.assertEqual(config["eloggen"]["trajectory_recipe"]["boundary_coverage_status"], "covered")


if __name__ == "__main__":
    unittest.main()
