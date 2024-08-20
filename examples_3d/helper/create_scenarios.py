from os.path import dirname, abspath

import numpy as np
from commonsky.geometry.shape import Cuboid, ShapeGroup, Bird
from commonsky.geometry.shape import TreeDeciduous
from commonsky.planning.goal import GoalRegion
from commonsky.planning.planning_problem import PlanningProblemSet, PlanningProblem
from commonsky.prediction.prediction import TrajectoryPrediction
from commonsky.scenario.obstacle import (
    StaticObstacle,
    ObstacleType,
    DynamicObstacle,
)
from commonsky.scenario.scenario import (
    Scenario,
)
from commonsky.scenario.state import InitialState3D, PMState3D, Interval
from commonsky.scenario.trajectory import Trajectory
from trimesh import PointCloud
from trimesh.collision import CollisionManager

from examples_3d.helper.create_obstacles import build_forrest, span_powerline, create_bird_swarm

DICT_PATH = dirname(dirname(dirname(abspath(__file__))))


# 1_DirectSimpleTrajectory
def create_direct_simple_trajectory(
    scenario: Scenario, planning_problem_set: PlanningProblemSet, random_seed=None
) -> None:
    """
    :param scenario: scenario where the obstacles are added
    :param planning_problem_set: panning problem set where the planning problem is added
    :param random_seed: random seed for reproducibility
    :return: None

    Creates the DirectSimpleTrajectory scenario
    """

    planning_problem = PlanningProblem(
        planning_problem_id=1,
        initial_state=InitialState3D(
            time_step=0,
            position=Cuboid(
                center=np.array([-5, 0, 2]),
                scaling=np.array([0.5] * 3),
            ),
            orientation=np.array([0] * 3),
            velocity=np.array([0] * 3),
        ),
        goal_region=GoalRegion(
            state_list=[
                PMState3D(
                    time_step=Interval(0, 100),
                    position=Cuboid(
                        center=np.array([5, 0, 2]),
                        scaling=np.array([0.5] * 3),
                    ),
                )
            ]
        ),
    )
    planning_problem_set.add_planning_problem(planning_problem)


# 2_SingleStaticObstacle
def create_single_static_obstacle(
    scenario: Scenario, planning_problem_set: PlanningProblemSet, random_seed=None
):
    """
    :param scenario: scenario where the obstacles are added
    :param planning_problem_set: panning problem set where the planning problem is added
    :param random_seed: random seed for reproducibility
    :return: None

    Creates the SingleStaticObstacle scenario
    """

    create_direct_simple_trajectory(scenario=scenario, planning_problem_set=planning_problem_set)
    static_obstacle = StaticObstacle(
        obstacle_id=1,
        obstacle_type=ObstacleType.TREE,
        obstacle_shape=TreeDeciduous(
            center=np.array([0, 0, 5]),
            scaling=np.array([5] * 3),
        ),
        initial_state=InitialState3D(),
    )
    scenario.add_objects(static_obstacle)


# 3_ThroughTheForrest
def create_through_the_forrest(
    scenario: Scenario, planning_problem_set: PlanningProblemSet, random_seed=None
):
    """
    :param scenario: scenario where the obstacles are added
    :param planning_problem_set: panning problem set where the planning problem is added
    :param random_seed: random seed for reproducibility
    :return: None

    Creates the ThroughTheForrest scenario
    """

    forrest = build_forrest(
        lower_x=-100,
        upper_x=100,
        lower_y=-100,
        upper_y=100,
        min_height=5,
        max_height=30,
        start_id=1,
        number_trees=500,
        threshold=200,
        random_seed=random_seed,
    )
    scenario.add_objects(forrest)

    planning_problem = PlanningProblem(
        planning_problem_id=1,
        initial_state=InitialState3D(
            time_step=0,
            position=Cuboid(
                center=np.array([-100, 0, 10]),
                scaling=np.array([1] * 3),
            ),
            orientation=np.array([0] * 3),
            velocity=np.array([0] * 3),
        ),
        goal_region=GoalRegion(
            state_list=[
                PMState3D(
                    time_step=Interval(0, 250),
                    position=Cuboid(
                        center=np.array([100, -9, 7]),
                        scaling=np.array([3] * 3),
                    ),
                )
            ]
        ),
    )
    planning_problem_set.add_planning_problem(planning_problem)


def create_basic_powerline(
    scenario: Scenario,
    planning_problem_set: PlanningProblemSet,
    cm: None | CollisionManager = None,
    random_seed=None,
):
    """
    :param scenario: scenario where the obstacles are added
    :param planning_problem_set: panning problem set where the planning problem is added
    :param cm: collision manager  for ensuring, that obstacles are not intersecting
    :param random_seed: random seed for reproducibility
    :return: None

    Creates the basis for powerline scenarios
    """

    if cm is None:
        cm = CollisionManager()

    # Powerline
    powerline = span_powerline(
        obstacle_id=1,
        start_x=0,
        start_y=-300,
        end_x=0,
        end_y=300,
        height=100,
        center_width=3,
        number_poles=7,
        number_sections=3,
        width_sections=50,
        cable_width=1,
    )
    powerline_shape = powerline.obstacle_shape
    assert isinstance(powerline_shape, ShapeGroup)

    for shape in powerline_shape.shapes:
        mesh = shape.apply_attributes()
        mesh = mesh.reshape(3 * mesh.shape[0], 3)
        mesh = PointCloud(mesh).convex_hull
        cm.add_object("mesh", mesh)
    scenario.add_objects(powerline)

    # Forrest
    scenario.add_objects(
        build_forrest(
            lower_x=-300,
            upper_x=300,
            lower_y=-300,
            upper_y=300,
            min_height=15,
            max_height=35,
            start_id=100,
            number_trees=900,
            threshold=200,
            collision_manager=cm,
            random_seed=random_seed,
        )
    )


# 4_GoingUpThePowerLine
def create_going_up_the_powerline(
    scenario: Scenario, planning_problem_set: PlanningProblemSet, random_seed=None
):
    """
    :param scenario: scenario where the obstacles are added
    :param planning_problem_set: panning problem set where the planning problem is added
    :param random_seed: random seed for reproducibility
    :return: None

    Creates the GoingUpThePowerLine scenario
    """

    create_basic_powerline(
        scenario=scenario,
        planning_problem_set=planning_problem_set,
        random_seed=random_seed,
    )
    planning_problem = PlanningProblem(
        planning_problem_id=1,
        initial_state=InitialState3D(
            time_step=0,
            position=Cuboid(
                center=np.array([10, -290, 2]),
                scaling=np.array([0.5] * 3),
            ),
            orientation=np.array([0] * 3),
            velocity=np.array([0] * 3),
        ),
        goal_region=GoalRegion(
            state_list=[
                PMState3D(
                    time_step=Interval(0, 200),
                    position=Cuboid(
                        center=np.array([10, -290, 90]),
                        scaling=np.array([3] * 3),
                    ),
                )
            ]
        ),
    )
    planning_problem_set.add_planning_problem(planning_problem)


# 5_InspectingPowerLine
def create_inspecting_powerline(
    scenario: Scenario, planning_problem_set: PlanningProblemSet, random_seed=None
):
    """
    :param scenario: scenario where the obstacles are added
    :param planning_problem_set: panning problem set where the planning problem is added
    :param random_seed: random seed for reproducibility
    :return: None

    Creates the InspectingPowerLine scenario
    """

    create_basic_powerline(
        scenario=scenario,
        planning_problem_set=planning_problem_set,
        random_seed=random_seed,
    )
    planning_problem = PlanningProblem(
        planning_problem_id=1,
        initial_state=InitialState3D(
            time_step=0,
            position=Cuboid(
                center=np.array([10, -290, 90]),
                scaling=np.array([1] * 3),
            ),
            orientation=np.array([0] * 3),
            velocity=np.array([0] * 3),
        ),
        goal_region=GoalRegion(
            state_list=[
                PMState3D(
                    time_step=Interval(0, 1000),
                    position=Cuboid(
                        center=np.array([-30, 290, 80]),
                        scaling=np.array([4] * 3),
                    ),
                )
            ]
        ),
    )
    planning_problem_set.add_planning_problem(planning_problem)


def create_extended_powerline(
    scenario: Scenario,
    planning_problem_set: PlanningProblemSet,
    cm: None | CollisionManager = None,
    random_seed=None,
):
    """
    :param scenario: scenario where the obstacles are added
    :param planning_problem_set: panning problem set where the planning problem is added
    :param cm: collision manager  for ensuring, that obstacles are not intersecting
    :param random_seed: random seed for reproducibility
    :return: None

    Extends the basic power line scenario to a more complex power line
    """

    if cm is None:
        cm = CollisionManager()
    create_basic_powerline(
        scenario=scenario,
        planning_problem_set=planning_problem_set,
        cm=cm,
        random_seed=random_seed,
    )
    # Extensions
    powerline = span_powerline(
        obstacle_id=2,
        start_x=0,
        start_y=300,
        end_x=0,
        end_y=900,
        height=100,
        center_width=3,
        number_poles=7,
        number_sections=3,
        width_sections=50,
        cable_width=1,
    )
    powerline_shape = powerline.obstacle_shape
    assert isinstance(powerline_shape, ShapeGroup)
    for shape in powerline_shape.shapes:
        mesh = shape.apply_attributes()
        mesh = mesh.reshape(3 * mesh.shape[0], 3)
        mesh = PointCloud(mesh).convex_hull
        cm.add_object("mesh", mesh)
    scenario.add_objects(powerline)

    powerline = span_powerline(
        obstacle_id=3,
        start_x=0,
        start_y=400,
        end_x=-300,
        end_y=400,
        height=85,
        center_width=3,
        number_poles=4,
        number_sections=3,
        width_sections=50,
        cable_width=1,
    )
    powerline_shape = powerline.obstacle_shape
    assert isinstance(powerline_shape, ShapeGroup)
    for shape in powerline_shape.shapes:
        mesh = shape.apply_attributes()
        mesh = mesh.reshape(3 * mesh.shape[0], 3)
        mesh = PointCloud(mesh).convex_hull
        cm.add_object("mesh", mesh)
    scenario.add_objects(powerline)

    powerline = span_powerline(
        obstacle_id=4,
        start_x=0,
        start_y=500,
        end_x=-300,
        end_y=700,
        height=85,
        center_width=3,
        number_poles=4,
        number_sections=2,
        width_sections=50,
        cable_width=1,
    )
    powerline_shape = powerline.obstacle_shape
    assert isinstance(powerline_shape, ShapeGroup)
    for shape in powerline_shape.shapes:
        mesh = shape.apply_attributes()
        mesh = mesh.reshape(3 * mesh.shape[0], 3)
        mesh = PointCloud(mesh).convex_hull
        cm.add_object("mesh", mesh)
    scenario.add_objects(powerline)

    powerline = span_powerline(
        obstacle_id=5,
        start_x=0,
        start_y=600,
        end_x=300,
        end_y=900,
        height=130,
        center_width=3,
        number_poles=5,
        number_sections=4,
        width_sections=50,
        cable_width=1,
    )
    powerline_shape = powerline.obstacle_shape
    assert isinstance(powerline_shape, ShapeGroup)
    for shape in powerline_shape.shapes:
        mesh = shape.apply_attributes()
        mesh = mesh.reshape(3 * mesh.shape[0], 3)
        mesh = PointCloud(mesh).convex_hull
        cm.add_object("mesh", mesh)
    scenario.add_objects(powerline)

    # Forrest
    scenario.add_objects(
        build_forrest(
            lower_x=-300,
            upper_x=300,
            lower_y=300,
            upper_y=900,
            min_height=15,
            max_height=35,
            start_id=1001,
            number_trees=1000,
            threshold=200,
            collision_manager=cm,
            random_seed=random_seed,
        )
    )


# 6_SplittingPowerLine
def create_splitting_powerline(
    scenario: Scenario, planning_problem_set: PlanningProblemSet, random_seed=None
):
    """
    :param scenario: scenario where the obstacles are added
    :param planning_problem_set: panning problem set where the planning problem is added
    :param random_seed: random seed for reproducibility
    :return: None

    Creates the SplittingPowerLine scenario
    """
    create_extended_powerline(
        scenario=scenario,
        planning_problem_set=planning_problem_set,
        random_seed=random_seed,
    )
    planning_problem = PlanningProblem(
        planning_problem_id=1,
        initial_state=InitialState3D(
            time_step=0,
            position=Cuboid(
                center=np.array([-30, 290, 80]),
                scaling=np.array([2] * 3),
            ),
            orientation=np.array([0] * 3),
            velocity=np.array([0] * 3),
        ),
        goal_region=GoalRegion(
            state_list=[
                PMState3D(
                    time_step=Interval(0, 1000),
                    position=Cuboid(
                        center=np.array([60, 900, 110]),
                        scaling=np.array([4] * 3),
                    ),
                )
            ]
        ),
    )
    planning_problem_set.add_planning_problem(planning_problem)


# 7_FlyingBackPowerline
def create_flying_back_powerline(
    scenario: Scenario, planning_problem_set: PlanningProblemSet, random_seed=None
):
    """
    :param scenario: scenario where the obstacles are added
    :param planning_problem_set: panning problem set where the planning problem is added
    :param random_seed: random seed for reproducibility
    :return: None

    Creates the FlyingBackPowerline scenario
    """
    create_extended_powerline(
        scenario=scenario,
        planning_problem_set=planning_problem_set,
        random_seed=random_seed,
    )
    planning_problem = PlanningProblem(
        planning_problem_id=1,
        initial_state=InitialState3D(
            time_step=0,
            position=Cuboid(
                center=np.array([60, 900, 110]),
                scaling=np.array([2] * 3),
            ),
            orientation=np.array([0] * 3),
            velocity=np.array([0] * 3),
        ),
        goal_region=GoalRegion(
            state_list=[
                PMState3D(
                    time_step=Interval(0, 2000),
                    position=Cuboid(
                        center=np.array([10, -290, 2]),
                        scaling=np.array([4] * 3),
                    ),
                )
            ]
        ),
    )
    planning_problem_set.add_planning_problem(planning_problem)


# 8_SingleDynamicObstacle
def create_single_dynamic_obstacle(
    scenario: Scenario, planning_problem_set: PlanningProblemSet, random_seed=None
):
    """
    :param scenario: scenario where the obstacles are added
    :param planning_problem_set: panning problem set where the planning problem is added
    :param random_seed: random seed for reproducibility
    :return: None

    Creates the SingleDynamicObstacle scenario
    """
    create_direct_simple_trajectory(
        scenario=scenario,
        planning_problem_set=planning_problem_set,
        random_seed=random_seed,
    )
    dynamic_obstacle = DynamicObstacle(
        obstacle_id=1,
        obstacle_type=ObstacleType.BIRD,
        obstacle_shape=Bird(),
        initial_state=InitialState3D(time_step=0),
        prediction=TrajectoryPrediction(
            trajectory=Trajectory(
                initial_time_step=0,
                state_list=[
                    PMState3D(
                        time_step=i,
                        position=np.array([0, i - 4, 2]),
                        velocity=np.array([0, 1, 0]),
                    )
                    for i in range(9)
                ],
            ),
            shape=Bird(),
        ),
    )
    scenario.add_objects(dynamic_obstacle)


# 9_FlyingThroughLinearBirdSwarm
def create_flying_trough_linear_bird_swarm(
    scenario: Scenario, planning_problem_set: PlanningProblemSet, random_seed=None
):
    """
    :param scenario: scenario where the obstacles are added
    :param planning_problem_set: panning problem set where the planning problem is added
    :param random_seed: random seed for reproducibility
    :return: None

    Creates the FlyingThroughLinearBirdSwarm scenario
    """
    create_basic_powerline(
        scenario=scenario,
        planning_problem_set=planning_problem_set,
        random_seed=random_seed,
    )
    bird_swarm = create_bird_swarm(
        f"{DICT_PATH}\\examples_3d\\bird_movements\\DataS2_mobbing_flock_#04_cropped.csv",
        offset=np.array([-13, 42, 33.5]),
        start_obstacle_id=2001,
    )
    scenario.add_objects(bird_swarm)

    planning_problem = PlanningProblem(
        planning_problem_id=1,
        initial_state=InitialState3D(
            time_step=0,
            position=Cuboid(
                center=np.array([10, 0, 80]),
                scaling=np.array([1] * 3),
            ),
            orientation=np.array([0] * 3),
            velocity=np.array([0] * 3),
        ),
        goal_region=GoalRegion(
            state_list=[
                PMState3D(
                    time_step=Interval(0, 200),
                    position=Cuboid(
                        center=np.array([-10, 60, 80]),
                        scaling=np.array([2] * 3),
                    ),
                )
            ],
        ),
    )
    planning_problem_set.add_planning_problem(planning_problem)


# 10_FlyingThroughTurningBirdSwarm
def create_flying_trough_turning_bird_swarm(
    scenario: Scenario, planning_problem_set: PlanningProblemSet, random_seed=None
):
    """
    :param scenario: scenario where the obstacles are added
    :param planning_problem_set: panning problem set where the planning problem is added
    :param random_seed: random seed for reproducibility
    :return: None

    Creates the FlyingThroughTurningBirdSwarm scenario
    """
    create_basic_powerline(
        scenario=scenario,
        planning_problem_set=planning_problem_set,
        random_seed=random_seed,
    )
    bird_swarm = create_bird_swarm(
        f"{DICT_PATH}\\examples_3d\\bird_movements\\DataS4_mobbing_flock_#04_cropped.csv",
        offset=np.array([5, 30, 80]),
        start_obstacle_id=2001,
    )
    scenario.add_objects(bird_swarm)

    planning_problem = PlanningProblem(
        planning_problem_id=1,
        initial_state=InitialState3D(
            time_step=0,
            position=Cuboid(
                center=np.array([10, 0, 80]),
                scaling=np.array([1] * 3),
            ),
            orientation=np.array([0] * 3),
            velocity=np.array([0] * 3),
        ),
        goal_region=GoalRegion(
            state_list=[
                PMState3D(
                    time_step=Interval(0, 200),
                    position=Cuboid(
                        center=np.array([-10, 60, 80]),
                        scaling=np.array([3] * 3),
                    ),
                )
            ],
        ),
    )
    planning_problem_set.add_planning_problem(planning_problem)
