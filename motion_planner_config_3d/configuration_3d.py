import numpy as np
from omegaconf import ListConfig, DictConfig
from commonsky.scenario.scenario import Scenario
from commonsky.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonsky.geometry.shape import Shape, UAV


class Configuration3D:
    """Base class holding all relevant configurations"""

    name_scenario: str
    scenario: None | Scenario
    planning_problem: None | PlanningProblem
    planning_problem_set: None | PlanningProblemSet

    def __init__(self, config: ListConfig | DictConfig):
        self.name_scenario = config.general.name_scenario
        self.scenario = None
        self.planning_problem = None
        self.planning_problem_set = None

        self.config_general = GeneralConfiguration(config)
        self.config_uav = UAVConfiguration(config)
        self.config_planning = PlanningConfiguration(config)

    @property
    def general(self) -> "GeneralConfiguration":
        return self.config_general

    @property
    def uav(self) -> "UAVConfiguration":
        return self.config_uav

    @property
    def planning(self) -> "PlanningConfiguration":
        return self.config_planning

    def complete_configuration(self, scenario, planning_problem, planning_problem_set) -> None:
        self.scenario = scenario
        self.planning_problem = planning_problem
        self.planning_problem_set = planning_problem_set
        self.uav.cr_vehicle_id = planning_problem.planning_problem_id


class GeneralConfiguration:
    """
    Configuration class for general attributes, like the scenario path
    """

    def __init__(self, config: ListConfig | DictConfig):
        config_general = config.get("general", dict())
        name_scenario = config_general.name_scenario

        self.path_root = config_general.path_root
        self.path_scenarios = config_general.path_scenarios
        self.path_scenario = config_general.path_scenarios + name_scenario + ".xml"
        self.path_output = config_general.path_output + name_scenario + "/"


class UAVConfiguration:
    """
    Configuration class for UAV attributes
    """

    uav_id: int
    v_min: float
    v_max: float
    a_min: float
    a_max: float
    radius: float
    shape: Shape

    def __init__(self, config: ListConfig | DictConfig):
        config_uav = config.get("uav", None)
        if config_uav is None:
            config_uav = dict()

        self.uav_id = config_uav.get("uav_id", 1)
        self.radius = config.get("radius", 1.0)
        self.v_min = config_uav.get("v_min", 0.0)
        self.v_max = config_uav.get("v_max", 0.0)
        self.a_min = config_uav.get("a_min", 0.0)
        self.a_max = config_uav.get("a_max", 0.0)
        self.shape = UAV(scaling=np.array([self.radius] * 3))

        for key, value in config_uav.items():
            if value is not None:
                setattr(self, key, value)


class PlanningConfiguration:
    """
    Configuration class for planning related attributes
    """

    dt: float
    time_start: float
    grid_size: float
    movement_cost_and_heuristic: str

    # fibonacci
    number_new_states_per_rad: int

    # RRT_Star
    total_states: int
    bounding_box: str

    def __init__(self, config: ListConfig | DictConfig):
        config_planning = config.get("planning", None)
        if config_planning is None:
            config_planning = dict()

        self.dt = config_planning.get("dt", 1.0)
        self.time_start = config_planning.get("time_start", 0.0)
        self.grid_size = config_planning.get("grid_size", 0.1)
        self.movement_cost_and_heuristic = config_planning.get(
            "movement_cost_and_heuristic", "velocity"
        )

        self.number_new_states_per_rad = config_planning.get("number_new_states_per_rad", 1)

        self.total_states = config_planning.get("total_states", 1000)
        self.bounding_box = config_planning.get("bounding_box", "all")

        self.motion_primitive = config_planning.get("motion_primitive", None)
        self.trajectory_planner = config_planning.get("trajectory_planner", None)
        self.predictor = config_planning.get("predictor", None)
