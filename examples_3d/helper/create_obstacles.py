import random
from random import uniform, choice

import numpy as np
import pandas as pd
from commonsky.geometry.shape import Cuboid, Cylinder, ShapeGroup, Bird
from commonsky.geometry.shape import Tree, TreeDeciduous
from commonsky.prediction.prediction import TrajectoryPrediction
from commonsky.scenario.obstacle import (
    StaticObstacle,
    ObstacleType,
    EnvironmentObstacle,
    DynamicObstacle,
)
from commonsky.scenario.state import InitialState3D, PMState3D
from commonsky.scenario.trajectory import Trajectory
from trimesh import PointCloud
from trimesh.collision import CollisionManager

TREE_TYPES = [Tree, TreeDeciduous]
MAX_BIRDS = 100


def build_forrest(
    lower_x,
    upper_x,
    lower_y,
    upper_y,
    min_height,
    max_height,
    start_id,
    number_trees,
    threshold,
    collision_manager: None | CollisionManager = None,
    random_seed=None,
) -> list[EnvironmentObstacle]:
    """
    :param lower_x: lower x bound for placing trees
    :param upper_x: upper x bound for placing trees
    :param lower_y: lower y bound for placing trees
    :param upper_y: upper y bound for placing trees
    :param min_height: minimum height of trees
    :param max_height: maximum height of trees
    :param start_id: id of the first tree
    :param number_trees: total number of trees that should be placed
    :param threshold: after this number of unsuccessful trys to place a tree the forrest is returned
    :param collision_manager: collision manager with obstacles trees should not intersect with
    :param random_seed: random seed for tree placement to ensure reproducibility
    :return: list of trees which do not intersect

    Place trees in an area to create a forrest
    """

    random.seed(random_seed)
    tree_shapes = []
    if collision_manager is None:
        cm = CollisionManager()
    else:
        cm = collision_manager
    for t in range(number_trees):
        for i in range(threshold):
            height = uniform(min_height, max_height)
            x = uniform(lower_x, upper_x)
            y = uniform(lower_y, upper_y)
            tree_type = choice(TREE_TYPES)
            new_tree = tree_type(
                center=np.array([x, y, height / 2]),
                orientation=np.array([0, 0, uniform(0, 2 * np.pi)]),
                scaling=np.array([height / 2, height / 2, height / 2]),
            )
            new_tree_mesh = new_tree.apply_attributes()
            new_tree_mesh_reshaped = new_tree_mesh.reshape(3 * new_tree_mesh.shape[0], 3)
            new_tree_tri_mesh = PointCloud(new_tree_mesh_reshaped).convex_hull
            if not cm.in_collision_single(new_tree_tri_mesh):
                cm.add_object("mesh", new_tree_tri_mesh)
                tree_shapes.append(new_tree)
                break
        else:
            break
    return [
        EnvironmentObstacle(
            obstacle_id=start_id + i,
            obstacle_type=ObstacleType.TREE,
            obstacle_shape=s,
        )
        for i, s in enumerate(tree_shapes)
    ]


def build_power_pole(
    center_x: float,
    center_y: float,
    height: float,
    orientation: np.ndarray,
    center_width: float,
    number_sections: int,
    width_section: float,
) -> ShapeGroup:
    """
    :param center_x: center x coordinate of the power pole
    :param center_y: center y coordinate of the power pole
    :param height:total height of the power pole
    :param orientation: direction the power pole is facing
    :param center_width: width of the power pole post itself
    :param number_sections: number of cross braces
    :param width_section: span width of the cross braces
    :return:ShapeGroup with all components of a power pole

    Creates a single power pole
    """

    middle = Cuboid(
        center=np.array([center_x, center_y, height / 2]),
        orientation=orientation,
        scaling=np.array([center_width / 2, center_width / 2, height / 2]),
    )

    assert 0 < number_sections, "need at least 1 section"
    sections = []
    if number_sections == 1:
        sections.append(
            Cuboid(
                center=np.array([center_x, center_y, height * 0.95]),
                orientation=orientation,
                scaling=np.array([width_section / 2, center_width / 2, center_width / 4]),
            )
        )
    elif 1 < number_sections:
        diff = ((height / 2) * 0.9) / (number_sections - 1)
        for i in range(number_sections):
            section_height = height / 2 + i * diff
            sections.append(
                Cuboid(
                    center=np.array([center_x, center_y, section_height]),
                    orientation=orientation,
                    scaling=np.array([width_section / 2, center_width / 2, center_width / 4]),
                )
            )

    return ShapeGroup(shapes=[middle] + sections)


def span_powerline(
    obstacle_id,
    start_x,
    start_y,
    end_x,
    end_y,
    height,
    center_width,
    number_poles: int,
    number_sections,
    width_sections,
    cable_width,
) -> StaticObstacle:
    """
    :param obstacle_id: obstacle id of the power line
    :param start_x: x coordinate of the first power pole
    :param start_y: y coordinate of the first power pole
    :param end_x: x coordinate of the last power pole
    :param end_y: y coordinate of the last power pole
    :param height:height of the power poles
    :param center_width:
    :param number_poles:total numbers of poles the power line has
    :param number_sections: number cross braces each power pole has
    :param width_sections: width of each cross brace
    :param cable_width: width of each cable
    :return:the whole power line as static obstacle

    Creates a straight power line from start to end position
    """

    assert 1 < number_poles, "need at least 2 poles"

    x = end_x - start_x
    y = end_y - start_y

    diff_x = x / (number_poles - 1)
    diff_y = y / (number_poles - 1)

    orientation = np.arctan2(y, x) + np.pi / 2

    # place poles
    poles = []
    for i in range(number_poles):
        pole: ShapeGroup = build_power_pole(
            center_x=start_x + i * diff_x,
            center_y=start_y + i * diff_y,
            height=height,
            orientation=np.array([0, 0, orientation]),
            center_width=center_width,
            number_sections=number_sections,
            width_section=width_sections,
        )
        poles.extend(pole.shapes)

    # place lines
    center_x = start_x + x / 2
    center_y = start_y + y / 2
    for i in range(number_sections):
        length = np.sqrt(x**2 + y**2) + center_width
        tran_x = (width_sections / 2 - cable_width / 2) * np.cos(orientation)
        tran_y = (width_sections / 2 - cable_width / 2) * np.sin(orientation)
        if number_sections == 1:
            section_height = height * 0.95 - center_width / 4
        elif 1 < number_sections:
            diff = ((height / 2) * 0.9) / (number_sections - 1)
            section_height = height / 2 + i * diff
        else:
            section_height = 0

        cable1 = Cylinder(
            center=np.array([center_x + tran_x, center_y + tran_y, section_height]),
            orientation=np.array([np.pi / 2, orientation, 0]),
            scaling=np.array([cable_width / 2, cable_width / 2, length / 2]),
        )
        cable2 = Cylinder(
            center=np.array([center_x - tran_x, center_y - tran_y, section_height]),
            orientation=np.array([np.pi / 2, orientation, 0]),
            scaling=np.array([cable_width / 2, cable_width / 2, length / 2]),
        )
        poles.append(cable1)
        poles.append(cable2)

    sg_poles = ShapeGroup(poles)

    return StaticObstacle(
        obstacle_id=obstacle_id,
        obstacle_type=ObstacleType.POWER_LINE,
        obstacle_shape=sg_poles,
        initial_state=InitialState3D(
            time_step=0,
            position=sg_poles.center,
        ),
    )


def create_bird_swarm(
    path: str,
    offset: None | np.ndarray = None,
    start_obstacle_id=1,
    max_birds=MAX_BIRDS,
) -> list[DynamicObstacle]:
    """
    :param path: path to the file here the bird swarm information are stored
    :param offset:vector by which the whole bird swarm is moved
    :param start_obstacle_id: obstacle id of the first bird
    :param max_birds: maximum amount of birds
    :return: list of birds as dynamic obstacles

    Reads trajectories of birds and converts them to dynamic obstacles
    """

    if offset is None:
        offset = np.array([0, 0, 0])
    df = pd.read_csv(
        filepath_or_buffer=path,
        sep=",",
    )
    t = {}
    for i, data in df.iterrows():
        id_ob = int(data["ID"])
        state = PMState3D(
            time_step=int(data["time_step"]),
            position=np.array([data["x(m)"], data["y(m)"], data["z(m)"]]) + offset,
            velocity=np.array([data["v_x(m/s)"], data["v_y(m/s)"], data["v_z(m/s)"]]),
        )
        if t.get(id_ob, None) is None:
            t[id_ob] = [state]
        else:
            t[id_ob].append(state)

    dym_obs = []
    for number, id_ in enumerate(list(t.keys())[:max_birds]):
        i = t[id_]
        traj = Trajectory(int(i[0].time_step), i)
        shape = Bird(scaling=np.array([1] * 3))
        dym_obs.append(
            DynamicObstacle(
                obstacle_id=int(start_obstacle_id + number),
                obstacle_type=ObstacleType.UNKNOWN,
                obstacle_shape=shape,
                initial_state=InitialState3D(time_step=int(i[0].time_step)),
                prediction=TrajectoryPrediction(trajectory=traj, shape=shape),
            )
        )

    return dym_obs
