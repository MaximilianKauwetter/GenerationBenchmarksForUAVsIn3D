import numpy as np
from commonsky.scenario.scenario import Scenario
from commonsky.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonsky.scenario.trajectory import Trajectory
from motion_planner_components_3d.trajectory_planner_3d.trajectory_planner_interface_3d import (
    TrajectoryPlannerInterface3D,
)
from motion_planner_components_3d.trajectory_planner_3d.A_star_3d import AStarSearch3D
from motion_planner_components_3d.trajectory_planner_3d.RRT_star_3d import RRTStarSearch3D
from motion_planner_components_3d.motion_primitives_3d.grid import GridMotionPrimitive
from motion_planner_components_3d.motion_primitives_3d.fibonacci_sphere import (
    FibonacciSphereMotionPrimitive,
)
from motion_planner_components_3d.prediction_3d.predictor_interface_3d import PredictorInterface3D
from motion_planner_config_3d.configuration_3d import Configuration3D
from commonsky.visualization.renderer import IRenderer
from commonsky.visualization.draw_params import O3DDrawParams3D
from commonsky.geometry.shape import UAV

MOTION_PRIMITIVES: dict[str, type] = {
    "grid": GridMotionPrimitive,
    "fibonacci_sphere": FibonacciSphereMotionPrimitive,
}
TRAJECTORY_PLANNERS: dict[str, type] = {
    "a_star": AStarSearch3D,
    "rrt_star": RRTStarSearch3D,
}


class MotionPlanner3D:
    """
    Motion Planner class that s the main interface for the motion planner library.
    It needs a Configuration3D object to get all kind of configuration details to find a trajectory for the UAV
    """

    config: Configuration3D
    scenario: Scenario
    planning_problem: PlanningProblem
    planning_problem_set: PlanningProblemSet

    trajectory_planner: None | TrajectoryPlannerInterface3D
    predictor: None | PredictorInterface3D

    planned_trajectory: None | Trajectory

    def __init__(self, config: Configuration3D):
        self.config = config
        self.scenario = config.scenario
        self.planning_problem = config.planning_problem
        self.planning_problem_set = config.planning_problem_set
        self.config.config_uav.shape = UAV(scaling=np.array([self.config.uav.radius] * 3))
        self.shape = self.config.config_uav.shape

        self._setup_trajectory_planner()
        self._setup_prediction_planner()

        self.planned_trajectory = None

    def _setup_trajectory_planner(self) -> None:
        """
        :return: None

        Sets up the trajectory planner according to the configuration
        """
        # create trajectory_planner
        trajectory_planner_type: type = TRAJECTORY_PLANNERS[
            self.config.config_planning.trajectory_planner
        ]
        self.trajectory_planner = trajectory_planner_type(self.config)

        # create motion_primitive for trajectory_planner
        motion_primitive_type = MOTION_PRIMITIVES[self.config.config_planning.motion_primitive]
        self.trajectory_planner.motion_primitive = motion_primitive_type(self.config)

    def _setup_prediction_planner(self):
        """
        :return: None

        Sets up the prediction planner according to the configuration
        """
        self.predictor = None

    def plan(self) -> Trajectory:
        """
        :return: None

        Uses the configured trajectory planner to find a trajectory
        """
        if self.predictor is not None:
            self.predictor.predict()

        self.planned_trajectory = self.trajectory_planner.plan()
        return self.planned_trajectory

    def draw(self, renderer: IRenderer, draw_params: None | O3DDrawParams3D = None):
        """
        :param renderer: Renderer that is used for visualization
        :param draw_params: draw params of the renderer
        :return: None

        Visualize the scenario, planning problem and planned trajectory with the renderer
        """

        self.scenario.draw(renderer=renderer, draw_params=draw_params)
        self.planning_problem.draw(
            renderer=renderer,
            draw_params=None if draw_params is None else draw_params.planning_problem,
        )

        if self.planned_trajectory is not None:
            if self.planned_trajectory.final_state.time_step < renderer.draw_params.time_begin:
                raise AttributeError(
                    f"draw_params.time_begin is bigger than the final_state time_stamp.\n"
                    f"\tdraw_params.time_begin: {renderer.draw_params.time_begin}\n"
                    f"\tfinal_state.time_stamp: {self.planned_trajectory.final_state.time_step}\n"
                )

            # history
            renderer.draw_history(
                trajectory=self.planned_trajectory,
                shape=self.shape,
                draw_params=None if draw_params is None else draw_params.history,
            )
            # at time_stamp
            state = self.planned_trajectory.state_at_time_step(renderer.draw_params.time_begin)
            vel = state.velocity
            self.shape.center = state.position
            self.shape.orientation = np.array([0, 0, np.arctan2(vel[1], vel[0]) + np.pi / 2])
            renderer.draw(
                vertices_mesh=self.shape.apply_attributes(),
                params=None if draw_params is None else draw_params.shape,
            )
            # future
            renderer.draw_trajectory(
                trajectory=self.planned_trajectory,
                shape=self.shape,
                draw_params=None if draw_params is None else draw_params.trajectory,
            )
