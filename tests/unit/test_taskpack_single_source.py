from pathlib import Path

from eloggen.generation_runtime.simulation.camera_config import (
    canonical_openarm_task_name,
    runtime_openarm_task_name,
)
from eloggen.pipeline.schema import PipelineConfig
from eloggen.pipeline.task_pack import discover_task_packs, load_task_pack


def test_discover_bundled_taskpacks():
    tasks = discover_task_packs()
    assert {
        "openarm_real_exp_1",
        "openarm_drawer_storage",
        "openarm_fruit_basket_bagging",
    }.issubset(tasks)


def test_runtime_task_name_resolves_back_to_taskpack():
    task = load_task_pack("openarm_real_exp_1_D1")
    assert task.name == "openarm_real_exp_1"
    assert canonical_openarm_task_name("openarm_real_exp_1_D1") == "openarm_real_exp_1"


def test_taskpack_declares_runtime_difficulty_and_generic_interface():
    task = load_task_pack("openarm_real_exp_1")
    assert task.default_difficulty == "D0"
    assert task.difficulty_variants == ("D0", "D1", "D2")
    assert task.processing["interface"] == "EG_TaskPackOmniGibsonInterface"
    assert task.runtime["compatibility"]["openarm_cashier_style"] is True
    assert runtime_openarm_task_name(task.name, "D2") == "openarm_real_exp_1_D2"


def test_pipeline_config_is_built_from_taskpack_paths():
    task = load_task_pack("openarm_real_exp_1")
    config = PipelineConfig.from_task_pack(task)

    assert config.task.name == task.name
    assert config.task.family == "object_container_storage"
    assert Path(config.task.base_config) == task.path("generation")
    assert Path(config.task.source_demo) == task.path("source")
    assert Path(config.task.processed_source_demo) == task.path("processed_source")
    assert Path(config.task.scene_file) == task.path("scene")
    assert config.generation.difficulty == "D0"
    assert config.logic.plan_id is None
    assert config.metadata["task_pack_driven"] is True


def test_drawer_articulated_runtime_definition_lives_in_manifest():
    task = load_task_pack("openarm_drawer_storage")
    interface = task.runtime["interface"]
    lower_drawer = interface["articulated_targets"]["lower_drawer"]

    assert lower_drawer["object_name"] == "drawer_cabinet_1"
    assert lower_drawer["joint_name"] == "j_link_1"
    assert lower_drawer["upper_limit"] > lower_drawer["lower_limit"]
