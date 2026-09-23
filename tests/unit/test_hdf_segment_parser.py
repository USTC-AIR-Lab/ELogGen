import unittest

from eloggen.pipeline.parsing import HDFSegmentParser
from eloggen.pipeline.replay_annotation import ReplayAnnotation, ReplayBoundary
from eloggen.planning.model import ExecutionPlan, InteractionSegment


CONFIG = "src/eloggen/datasets/taskpacks/openarm_real_exp_1/generation.json"
HDF = "src/eloggen/datasets/taskpacks/openarm_real_exp_1/source/processed.hdf5"


class HDFSegmentParserTests(unittest.TestCase):
    def test_real_exp1_yields_four_lazy_interaction_segments(self):
        parsed = HDFSegmentParser(CONFIG, HDF).parse()
        self.assertEqual(len(parsed.segments), 4)
        self.assertEqual(
            [(item.alpha, item.boundary.start_frame, item.boundary.end_frame) for item in parsed.segments],
            [("pick", 210, 420), ("place", 500, 645), ("pick", 900, 1110), ("place", 1230, 1350)],
        )
        self.assertEqual({item.roles[0].entity for item in parsed.segments}, {"object_1", "object_2"})
        self.assertIn("datagen_info/gripper_action", parsed.source_manifest["dataset_keys"])
        self.assertEqual(
            [item.metadata["gripper_transition_frames"] for item in parsed.segments],
            [[323], [568], [1000], [1275]],
        )
        self.assertTrue(all(item.metadata["boundary_verified_by_gripper"] for item in parsed.segments))
        self.assertFalse(hasattr(parsed.segments[0].source_slice, "action"))

    def test_manual_annotation_overrides_config_boundary(self):
        annotation = ReplayAnnotation(
            demo_id="demo_0",
            boundaries=[
                ReplayBoundary("s", "pick_place.grasp_object", "grasp", "replay_start", 220, object="object_2", effector="left"),
                ReplayBoundary("e", "pick_place.grasp_object", "grasp", "replay_end", 410, object="object_2", effector="left"),
            ],
        )
        parsed = HDFSegmentParser(CONFIG, HDF).parse(annotation)
        lemon_pick = next(item for item in parsed.segments if item.id.startswith("phase_1"))
        self.assertEqual((lemon_pick.boundary.start_frame, lemon_pick.boundary.end_frame), (220, 410))
        self.assertEqual(lemon_pick.metadata["boundary_source"], "annotation")

    def test_new_schema_round_trip_and_old_plan_compatibility(self):
        segment = HDFSegmentParser(CONFIG, HDF).parse().segments[0]
        self.assertEqual(InteractionSegment.from_dict(segment.to_dict()), segment)
        old_plan = ExecutionPlan.from_dict({"id": "legacy", "stage_sequence": []})
        self.assertEqual(old_plan.coordination_mode, "sequential")
        self.assertEqual(old_plan.role_bindings, {})


if __name__ == "__main__":
    unittest.main()
