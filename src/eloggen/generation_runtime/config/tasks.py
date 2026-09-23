"""Generation defaults for the three bundled OpenArm tasks."""

from eloggen.generation_runtime.config.base import EG_Config


class EG_OpenArmFruitBasketBagging(EG_Config):
    NAME = "openarm_fruit_basket_bagging"
    TYPE = "omnigibson_bimanual"

    def task_config(self):
        self.task.task_spec.phase1 = dict()
        self.task.task_spec.do_not_lock_keys()
        self.task.initialization_distribution.name = "D0"
        self.task.initialization_distribution.difficulty = "D0"
        self.task.initialization_distribution.object_xy_magnitude = 0.04
        self.task.initialization_distribution.object_yaw_deg = 0.0
        self.task.initialization_distribution.container_x_magnitude = 0.0
        self.task.initialization_distribution.container_y_magnitude = 0.0
        self.task.initialization_distribution.container_yaw_deg = 0.0
        self.task.initialization_distribution.robot_arm_joint_magnitude_deg = 0.0
        self.task.initialization_distribution.object_pair_clearance_m = 0.05
        self.task.initialization_distribution.object_container_clearance_m = 0.05
        self.task.initialization_distribution.container_pair_clearance_m = 0.08
        self.task.initialization_distribution.object_container_repair_gap_m = 0.05
        self.task.initialization_distribution.preserve_left_to_right_order = True
        self.task.initialization_distribution.allow_object_order_permutation = False
        self.task.initialization_distribution.max_sampling_attempts = 50
        self.fruit_source.trim_end_exclusive = None
        self.fruit_source.source_order = []
        self.fruit_source.boundaries = []
        self.subtask_graph.nodes = []
        self.subtask_graph.edges = []
        self.subtask_graph.atomic_chains = []
        self.subtask_graph.resource_constraints = {}
        self.subtask_graph.resource_events = {}
        self.subtask_scheduling.strategy = "random_topological"
        self.subtask_scheduling.graph_source = "auto"
        self.subtask_scheduling.fixed_order = []
        self.subtask_scheduling.legal_order_names = []


class EG_OpenArmRealExp1(EG_Config):
    NAME = "openarm_real_exp_1"
    TYPE = "omnigibson_bimanual"

    def task_config(self):
        self.task.task_spec.phase1 = dict()
        self.task.task_spec.do_not_lock_keys()


class EG_OpenArmDrawerStorage(EG_Config):
    NAME = "openarm_drawer_storage"
    TYPE = "omnigibson_bimanual"

    def task_config(self):
        self.task.task_spec.phase1 = dict()
        self.task.task_spec.do_not_lock_keys()
        self.task.initialization_distribution.name = "D0"
        self.task.initialization_distribution.difficulty = "D0"
        self.task.initialization_distribution.object_xy_magnitude = 0.02
        self.task.initialization_distribution.object_yaw_deg = 0.0
        self.task.initialization_distribution.cabinet_x_magnitude = 0.0
        self.task.initialization_distribution.cabinet_y_magnitude = 0.0
        self.task.initialization_distribution.cabinet_yaw_deg = 0.0
        self.task.initialization_distribution.robot_arm_joint_magnitude_deg = 0.0
        self.task.initialization_distribution.drawer_open_distance_m = 0.184
        self.task.initialization_distribution.object_drawer_clearance_m = 0.08
        self.task.initialization_distribution.object_pair_clearance_m = 0.08
        self.task.initialization_distribution.drawer_open_fraction = [0.0, 0.0]
        self.task.initialization_distribution.max_sampling_attempts = 20
        self.drawer_source.trim_end_exclusive = None
        self.drawer_source.source_order = []
        self.drawer_source.boundaries = []
        self.subtask_graph.nodes = []
        self.subtask_graph.edges = []
        self.subtask_graph.atomic_chains = []
        self.subtask_graph.resource_constraints = {}
        self.subtask_graph.resource_events = {}
        self.subtask_scheduling.strategy = "fixed"
        self.subtask_scheduling.graph_source = "auto"
        self.subtask_scheduling.fixed_order = []
        self.subtask_scheduling.legal_order_names = []
