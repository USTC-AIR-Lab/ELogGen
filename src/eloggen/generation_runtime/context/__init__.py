"""Utilities for stage boundary detection and context injection."""

from eloggen.generation_runtime.context.actions import ACTION_SPECS, get_action_spec, supported_actions

__all__ = [
    "ACTION_SPECS",
    "annotate_stage_boundaries",
    "build_generated_segment_contexts",
    "get_action_spec",
    "supported_actions",
]


def __getattr__(name):
    if name == "annotate_stage_boundaries":
        from eloggen.generation_runtime.context.boundary import annotate_stage_boundaries

        return annotate_stage_boundaries
    if name == "build_generated_segment_contexts":
        from eloggen.generation_runtime.context.frames import build_generated_segment_contexts

        return build_generated_segment_contexts
    raise AttributeError(name)
