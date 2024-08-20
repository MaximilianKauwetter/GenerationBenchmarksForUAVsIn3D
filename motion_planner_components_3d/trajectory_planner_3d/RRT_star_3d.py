import logging
import random
from datetime import datetime

import numpy as np
from commonsky.geometry.shape import Shape
from commonsky.scenario.obstacle import (
    StaticObstacle,
    EnvironmentObstacle,
    DynamicObstacle,
    PhantomObstacle,
)
from commonsky.scenario.state import TraceState, PMState3D
from rtree.index import Index, Property

from collision_checker_3d.collision_checker_3d import CollisionChecker3D
from motion_planner_components_3d.trajectory_planner_3d.trajectory_planner_interface_3d import (
    TrajectoryPlannerInterface3D,
)
from motion_planner_config_3d.configuration_3d import Configuration3D

RANDOM_SEED = 42

logger = logging.Logger(name="RRT_Star")
logger.setLevel(logging.NOTSET)
logger.addHandler(logging.StreamHandler())


class RRTStarSearch3D(TrajectoryPlannerInterface3D):
    """
    Trajectory planner using the RRT* path planning algorithm

    """

    def __init__(self, config: Configuration3D):
        logger.debug("Init RRT_Star")
        time_start = datetime.now()
        super().__init__(config)
        self.grid_size = config.config_planning.grid_size
        self.number_nodes = config.config_planning.total_states
        self.min_velocity = config.config_uav.v_min
        self.max_velocity = config.config_uav.v_max
        self.min_acceleration = config.config_uav.a_min
        self.max_acceleration = config.config_uav.a_max

        self.list_obstacles = self.scenario.obstacles
        self.uav_shape = self.config.config_uav.shape
        self.collision_checker = CollisionChecker3D(
            uav_shape=self.uav_shape,
            obstacles=self.list_obstacles,
        )

        self.movement_cost = {
            "euc_distance": lambda velocity: velocity,
            "velocity": lambda velocity: 1,
            "max_velocity": lambda position_3d, velocity: 1,
        }[config.config_planning.movement_cost_and_heuristic]
        self.bounding_box = config.config_planning.bounding_box
        self.min_x = 0
        self.max_x = 0
        self.grid_x = []
        self.min_y = 0
        self.max_y = 0
        self.grid_y = []
        self.min_z = 0
        self.max_z = 0
        self.grid_z = []
        self.set_search_space_limits()

        time_end = datetime.now()
        self.time_init = time_end - time_start
        logger.debug(f"Inited RRT_Star: {time_end-time_start}")

    def set_search_space_limits(self):
        """
        :return: None

        Sets the bounding box according to which objects need to be contained
        """
        all_positions = [self.state_initial.position.center, self.goal_coordinates]
        all_radii = [0]

        # add all obstacle positions and radii
        if self.bounding_box in ["all", "static", "dynamic"]:
            static_obstacles: list[StaticObstacle | EnvironmentObstacle] = []
            dynamic_obstacles: list[DynamicObstacle | PhantomObstacle] = []
            for obstacle in self.list_obstacles:
                if isinstance(obstacle, (StaticObstacle, EnvironmentObstacle)):
                    static_obstacles.append(obstacle)
                if isinstance(obstacle, (DynamicObstacle, PhantomObstacle)):
                    dynamic_obstacles.append(obstacle)
            if self.bounding_box in ["all", "static"]:
                for static_obstacle in static_obstacles:
                    all_positions.append(static_obstacle.obstacle_shape.center)
                    all_radii.append(CollisionChecker3D.calc_radius(static_obstacle.obstacle_shape))
            if self.bounding_box in ["all", "dynamic"]:
                for dynamic_obstacle in dynamic_obstacles:
                    for state in dynamic_obstacle.prediction.trajectory.state_list:
                        position = state.position
                        if isinstance(position, np.ndarray):
                            all_positions.append(position)
                        elif isinstance(position, Shape):
                            all_positions.append(position.center)
                all_radii.extend(
                    [
                        dynamic_obstacle[1]
                        for dynamic_obstacle in self.collision_checker.dynamic_obstacles
                    ]
                )

        # set margin for ensuring, that the drone could fly around all obstacles
        margin = max(all_radii) + self.collision_checker.uav_radius + self.grid_size

        # tes bounding box limits
        x_values = [pos[0] for pos in all_positions]
        self.min_x = min(x_values) - margin
        self.max_x = max(x_values) + margin

        y_values = [pos[1] for pos in all_positions]
        self.min_y = min(y_values) - margin
        self.max_y = max(y_values) + margin

        z_values = [pos[2] for pos in all_positions]
        self.min_z = min(z_values) - margin
        self.max_z = max(z_values) + margin

        # creating grid
        self.grid_x = np.arange(start=self.min_x, stop=self.max_x, step=self.grid_size)
        self.grid_y = np.arange(start=self.min_y, stop=self.max_y, step=self.grid_size)
        self.grid_z = np.arange(start=self.min_z, stop=self.max_z, step=self.grid_size)

    def execute_search(self) -> None | list[TraceState]:
        """
        :return: path as list of states or None if no path was found

        Implementation of the RRT* algorithm
        """
        random.seed(RANDOM_SEED)
        logger.debug("Execute RRT_Star")
        time_start = datetime.now()

        tree = Index(
            bounds=(self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z),
            properties=Property(dimension=3),
        )

        # insert initial state
        initial_state = self.state_initial
        position_3d = initial_state.position.center
        tree.insert(
            id=0,
            coordinates=np.tile(position_3d, 2),
            obj={
                "time_step": initial_state.time_step,
                "position_3d": position_3d,
                "velocity_3d": initial_state.velocity,
                "velocity": np.linalg.norm(initial_state.velocity),
                "costs": 0,
                "predecessor": None,
            },
        )

        # goal node
        goal_node = None

        for id_node in range(1, self.number_nodes):
            # new random node
            position_x = random.choice(self.grid_x)
            position_y = random.choice(self.grid_y)
            position_z = random.choice(self.grid_z)
            position_3d = np.array([position_x, position_y, position_z])
            # check if already in tree
            if tree.count(np.tile(position_3d, 2)) != 0:
                continue

            neighbours = []
            possibilities = []
            # check neighbours on reachability
            for neighbour in tree.intersection(
                coordinates=(
                    position_x - self.max_velocity,
                    position_y - self.max_velocity,
                    position_z - self.max_velocity,
                    position_x + self.max_velocity,
                    position_y + self.max_velocity,
                    position_z + self.max_velocity,
                ),
                objects=True,
            ):
                neighbour_obj = neighbour.object
                neighbour_velocity = neighbour_obj["velocity"]
                neighbour_difference_3d = neighbour_obj["position_3d"] - position_3d
                neighbour_difference = np.linalg.norm(neighbour_difference_3d)
                if neighbour_difference <= self.max_velocity:
                    neighbour_node = {
                        "id": neighbour.id,
                        **neighbour_obj,
                        "difference_3d": neighbour_difference_3d,
                        "difference": neighbour_difference,
                    }
                    neighbours.append(neighbour_node)
                    time_step = neighbour_obj["time_step"] + 1
                    min_neighbour_velocity = max(
                        self.min_velocity, neighbour_velocity + self.min_acceleration
                    )
                    max_neighbour_velocity = min(
                        self.max_velocity, neighbour_velocity + self.max_acceleration
                    )
                    if (
                        min_neighbour_velocity <= neighbour_difference <= max_neighbour_velocity
                        and not self.collision_checker.check_collision(
                            position=position_3d,
                            timestep=time_step,
                        )
                    ):
                        possibility = {
                            "time_step": time_step,
                            "position_3d": position_3d,
                            "velocity_3d": neighbour_difference_3d,
                            "velocity": neighbour_difference,
                            "costs": neighbour_obj["costs"]
                            + self.movement_cost(neighbour_difference),
                            "predecessor": neighbour.id,
                        }
                        possibilities.append(possibility)
            if len(possibilities) == 0:
                # no neighbour which can be directly connected
                nearest = list(tree.nearest(coordinates=position_3d, num_results=1, objects=True))[
                    0
                ]
                nearest_obj = nearest.object
                nearest_time_step = nearest_obj["time_step"]
                nearest_position_3d = nearest_obj["position_3d"]
                nearest_velocity = nearest_obj["velocity"]
                difference_3d = position_3d - nearest_position_3d
                difference = np.linalg.norm(difference_3d)

                min_nearest_velocity = max(
                    self.min_velocity, nearest_velocity + self.min_acceleration
                )
                max_nearest_velocity = min(
                    self.max_velocity, nearest_velocity + self.max_acceleration
                )
                if difference < min_nearest_velocity:
                    # if nearest is too close
                    new_velocity = min_nearest_velocity
                    round_func = lambda mov: np.copysign(np.ceil(np.abs(mov)), mov)
                elif max_nearest_velocity < difference:
                    # if nearest is too far
                    new_velocity = max_nearest_velocity
                    round_func = np.trunc
                else:
                    continue

                movement = (difference_3d / difference) * new_velocity
                movement = round_func(movement / self.grid_size) * self.grid_size
                velocity = np.linalg.norm(movement)
                position_3d = nearest_position_3d + movement
                position_3d = np.round(position_3d / self.grid_size) * self.grid_size
                time_step = nearest_time_step + 1

                if self.collision_checker.check_collision(
                    position=position_3d,
                    timestep=time_step,
                ):
                    continue

                new_node = {
                    "time_step": time_step,
                    "position_3d": position_3d,
                    "velocity_3d": movement,
                    "velocity": velocity,
                    "costs": nearest_obj["costs"] + self.movement_cost(velocity),
                    "predecessor": nearest.id,
                }
            else:
                # connect to neighbour with the lowest total cost
                new_node = min(possibilities, key=lambda p: p["costs"])
            # insert new node
            tree.insert(
                id=id_node,
                coordinates=np.tile(position_3d, 2),
                obj=new_node,
            )
            # check if goal could be reached
            max_new_velocity = min(self.max_velocity, new_node["velocity"] + self.max_acceleration)
            distance = self.calc_euclidean_distance(position=new_node["position_3d"])
            time_step = new_node["time_step"] + 1
            if distance <= max_new_velocity and self.time_desired.contains(time_step):
                costs = new_node["costs"] + self.movement_cost(distance)
                if goal_node is None or costs < goal_node["costs"]:
                    goal_node = {
                        "time_step": time_step,
                        "position_3d": self.goal_coordinates,
                        "velocity_3d": self.goal_coordinates - new_node["position_3d"],
                        "velocity": distance,
                        "costs": costs,
                        "predecessor": id_node,
                    }

            # rewire neighbours if costs would get decreased
            velocity = new_node["velocity"]
            min_velocity = max(self.min_velocity, velocity + self.min_acceleration)
            max_velocity = min(self.max_velocity, velocity + self.max_acceleration)
            for neighbour in neighbours:
                same_velocity = np.isclose(neighbour["difference"], neighbour["velocity"])
                nearly_max_velocity = (
                    self.max_velocity + self.min_acceleration
                    <= neighbour["difference"]
                    <= neighbour["velocity"]
                )
                nearly_min_velocity = (
                    neighbour["velocity"]
                    <= neighbour["difference"]
                    <= self.min_velocity + self.max_acceleration
                )
                is_reachable = min_velocity <= neighbour["difference"] <= max_velocity
                if is_reachable and (nearly_min_velocity or same_velocity or nearly_max_velocity):
                    costs = new_node["costs"] + self.movement_cost(neighbour["difference"])
                    if costs < neighbour["costs"]:
                        # Rewire
                        coordinates = np.tile(neighbour["position_3d"], 2)
                        tree.delete(
                            id=neighbour["id"],
                            coordinates=coordinates,
                        )
                        tree.insert(
                            id=neighbour["id"],
                            coordinates=coordinates,
                            obj={
                                "time_step": new_node["time_step"] + 1,
                                "position_3d": neighbour["position_3d"],
                                "velocity_3d": neighbour["difference_3d"],
                                "velocity": neighbour["difference"],
                                "costs": costs,
                                "predecessor": id_node,
                            },
                        )
        if goal_node is None:
            # return None if the goal node was not set
            state_list = None
        else:
            # return the path if the goal node is set
            state_list = []
            all_nodes = {
                node.id: node.object
                for node in tree.intersection(coordinates=tree.bounds, objects=True)
            }
            node = goal_node
            # iterate through predecessor till initial state is reached
            while node is not None:
                state_list.insert(
                    0,
                    PMState3D(
                        time_step=node["time_step"],
                        position=node["position_3d"],
                        velocity=node["velocity_3d"],
                    ),
                )
                node = all_nodes.get(node["predecessor"], None)
        time_end = datetime.now()
        self.time_execute = time_end - time_start
        if state_list is not None:
            self.costs = goal_node["costs"]
            logger.debug(
                f"Executed RRT_Star.\n"
                f"\tPath found: True\n"
                f"\tElapsed time: {self.time_execute}\n"
                f"\tTotal costs: {self.costs}\n"
                f"\tNumber states: {len(state_list)}\n"
            )
        else:
            logger.debug(
                f"Executed RRT_Star.\n"
                f"\tPath found: False\n"
                f"\tElapsed time: {self.time_execute}\n"
            )
        return state_list
