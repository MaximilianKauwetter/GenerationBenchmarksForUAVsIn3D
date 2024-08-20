from abc import ABC, abstractmethod
import numpy as np
from commonsky.geometry.shape import Shape, Cuboid
from commonsky.scenario.scenario import Scenario
from commonsky.scenario.state import TraceState
from commonsky.scenario.trajectory import Trajectory
from motion_planner_config_3d.configuration_3d import Configuration3D
from motion_planner_components_3d.motion_primitives_3d.motion_primitive_interface_3d import (
    MotionPrimitiveInterface3D,
)
from collision_checker_3d.collision_checker_3d import CollisionChecker3D
from commonsky.common.util import Interval
from commonsky.planning.planning_problem import PlanningProblem
from commonsky.scenario.state import InitialState3D


class TrajectoryPlannerInterface3D(ABC):
    """
    Base class for trajectory planner.
    """

    config: Configuration3D
    scenario: Scenario
    planning_problem: PlanningProblem
    motion_primitive: None | MotionPrimitiveInterface3D
    collision_checker: None | CollisionChecker3D

    def __init__(self, config: Configuration3D):
        self.config = config
        self.scenario = config.scenario
        self.planning_problem = config.planning_problem
        self.state_initial: InitialState3D = self.planning_problem.initial_state
        self.goal_coordinates = None
        self.time_desired = None
        self.position_desired = None
        self.velocity_desired = None
        self.orientation_desired = None
        self.parse_planning_problem()

    def plan(self) -> None | Trajectory:
        """
        :return: Trajectory or None if no Trajectory was found

        Executes the path planning and converts the result into a Trajectory if possible
        """
        state_list: None | list[TraceState] = self.execute_search()
        if state_list is None:
            return None
        initial_time_step = state_list[0].time_step

        return Trajectory(
            initial_time_step=initial_time_step,
            state_list=state_list,
        )

    @abstractmethod
    def execute_search(self) -> None | list[TraceState]:
        """
        :return: path as list of states or None if no path is found

        Executes the implemented path-planing algorithm
        """
        pass

    def parse_planning_problem(self) -> None:
        """
        Parses the given planning problem, and computes related attributes.
        """
        assert isinstance(
            self.planning_problem, PlanningProblem
        ), "Given planning problem is not valid!"
        goal: TraceState = self.planning_problem.goal.state_list[0]

        # set attributes with given planning problem
        self.time_desired: Interval = goal.time_step

        self.position_desired = None
        if hasattr(goal, "position"):
            if isinstance(goal.position, Cuboid):
                center_x, center_y, center_z = goal.position.center
                scaling_x, scaling_y, scaling_z = goal.position.scaling
                self.position_desired = [
                    Interval(start=center_x - scaling_x, end=center_x + scaling_x),
                    Interval(start=center_y - scaling_y, end=center_y + scaling_y),
                    Interval(start=center_z - scaling_z, end=center_z + scaling_z),
                ]
                self.goal_coordinates = goal.position.center
            else:
                if isinstance(goal.position, np.ndarray):
                    center_x, center_y, center_z = goal.position
                elif isinstance(goal.position, Shape):
                    center_x, center_y, center_z = goal.position.center
                else:
                    center_x, center_y, center_z = 0, 0, 0

                self.position_desired = [
                    Interval(start=center_x, end=center_x),
                    Interval(start=center_y, end=center_y),
                    Interval(start=center_z, end=center_z),
                ]
                self.goal_coordinates = np.array([center_x, center_y, center_z])

        if hasattr(goal, "velocity"):
            self.velocity_desired = goal.velocity
        else:
            self.velocity_desired = Interval(0, np.inf)

    def calc_euclidean_distance(self, position: np.ndarray) -> float:
        """
        :param position: current position as 3d coordinates
        :return: euclidean distance between position and goal region

        Calculates the euclidean distance to the desired goal position.
        """
        delta = [
            0.0
            if self.position_desired[i].contains(axis)
            else min(
                [
                    abs(self.position_desired[i].start - axis),
                    abs(self.position_desired[i].end - axis),
                ]
            )
            for i, axis in enumerate(position)
        ]
        return np.linalg.norm(delta)
