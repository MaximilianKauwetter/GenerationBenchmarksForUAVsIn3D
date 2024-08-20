import numpy as np
from datetime import datetime

from commonsky.scenario.state import TraceState, PMState3D
from queue import PriorityQueue
from motion_planner_config_3d.configuration_3d import Configuration3D
from motion_planner_components_3d.trajectory_planner_3d.trajectory_planner_interface_3d import (
    TrajectoryPlannerInterface3D,
)
from collision_checker_3d.collision_checker_3d import CollisionChecker3D
import logging

logger = logging.Logger(name="A_Star")
logger.setLevel(logging.NOTSET)
logger.addHandler(logging.StreamHandler())


class AStarSearch3D(TrajectoryPlannerInterface3D):
    """
    Trajectory Planner class using the A* path planning algorithm.
    """

    def __init__(self, config: Configuration3D):
        logger.debug("Init A_Star")
        time_start = datetime.now()
        super().__init__(config)
        self.list_obstacles = self.scenario.obstacles
        self.uav_shape = self.config.config_uav.shape
        self.collision_checker = CollisionChecker3D(
            uav_shape=self.uav_shape,
            obstacles=self.list_obstacles,
        )
        self.movement_cost_and_heuristic_function = {
            "euc_distance": lambda position_3d, velocity: (
                velocity,
                self.calc_euclidean_distance(position_3d),
            ),
            "velocity": lambda position_3d, velocity: (
                1,
                np.inf
                if np.isclose(velocity, 0)
                else self.calc_euclidean_distance(position_3d) / velocity,
            ),
            "max_velocity": lambda position_3d, velocity: (
                1,
                self.calc_euclidean_distance(position_3d) / self.config.config_uav.v_max,
            ),
        }[config.config_planning.movement_cost_and_heuristic]

        time_end = datetime.now()
        self.time_init = time_end - time_start
        logger.debug("Inited A_Star")
        logger.debug(f"Elapsed time: {self.time_init}")

    def execute_search(self) -> None | list[TraceState]:
        """
        :return: path as list of states or None if no path was found

        Implementation of the A* algorithm
        """
        logger.info("Execute A_Star search")
        time_start = datetime.now()
        open_list = PriorityQueue()
        close_list: list[TraceState] = []
        observed_set: set = set()

        # add initial state
        initial_state = self.state_initial
        state_counter = 0
        position_3d = initial_state.position.center
        number = state_counter
        time_step = initial_state.time_step
        velocity_3d = initial_state.velocity
        velocity = np.linalg.norm(velocity_3d)
        costs, heuristic = self.movement_cost_and_heuristic_function(
            position_3d=position_3d,
            velocity=velocity,
        )
        costs = 0
        approximation = heuristic
        predecessors = []
        open_list.put(
            (
                approximation,
                position_3d[0],
                position_3d[1],
                position_3d[2],
                -velocity,
                approximation,
                number,
                time_step,
                velocity_3d,
                predecessors,
                costs,
            )
        )
        observed_set.add((*position_3d, velocity))

        while not open_list.empty():
            # take state with the lowest approximation
            (
                approximation,
                position_x,
                position_y,
                position_z,
                velocity,
                heuristic,
                number,
                time_step,
                velocity_3d,
                predecessors,
                costs,
            ) = open_list.get()
            position_3d = np.array([position_x, position_y, position_z])
            velocity = -velocity

            state = PMState3D(
                time_step=time_step,
                position=position_3d,
                velocity=velocity_3d,
            )

            # check collisions
            if self.collision_checker.check_collision(
                position=position_3d,
                timestep=time_step,
            ):
                continue

            index_state = len(close_list)
            close_list.append(state)

            # get successor states
            successors = self.motion_primitive.calculate_successor(
                position=position_3d,
                velocity=velocity,
                goal_position=self.goal_coordinates,
            )
            # enqueue all new successor states
            for successor_position_3d, successor_velocity_3d in successors:
                successor_time_step = time_step + 1
                if not self.time_desired.contains(successor_time_step):
                    continue
                successor_velocity = np.linalg.norm(successor_velocity_3d)
                node_hash = (*successor_position_3d, successor_velocity)
                if node_hash in observed_set:
                    continue
                else:
                    observed_set.add(node_hash)

                state_counter += 1
                (
                    successor_movement_costs,
                    successor_heuristic,
                ) = self.movement_cost_and_heuristic_function(
                    position_3d=successor_position_3d,
                    velocity=successor_velocity,
                )
                successor_costs = costs + successor_movement_costs
                successor_approximation = successor_costs + successor_heuristic
                successor_predecessors = predecessors + [index_state]
                successor_costs = costs + successor_movement_costs

                # check if successor is in goal
                if np.isclose(successor_heuristic, 0):
                    # if successor is in goal return total path
                    state_list = [close_list[i] for i in successor_predecessors] + [
                        PMState3D(
                            time_step=successor_time_step,
                            position=successor_position_3d,
                            velocity=successor_velocity_3d,
                        )
                    ]
                    time_end = datetime.now()
                    self.time_execute = time_end - time_start
                    self.costs = successor_costs
                    logger.info(
                        f"Executed A_Star.\n"
                        f"\tPath found: True\n"
                        f"\tElapsed time: {self.time_execute}\n"
                        f"\tTotal costs: {self.costs}\n"
                        f"\tFinal timestep: {state_list[-1].time_step}\n"
                        f"\tNumber states: {len(state_list)}\n"
                    )
                    return state_list

                successor_node = (
                    successor_approximation,
                    successor_position_3d[0],
                    successor_position_3d[1],
                    successor_position_3d[2],
                    -successor_velocity,
                    successor_heuristic,
                    state_counter,
                    successor_time_step,
                    successor_velocity_3d,
                    successor_predecessors,
                    successor_costs,
                )
                # enqueue successor state
                try:
                    open_list.put(successor_node)
                except Exception as e:
                    logger.warning(f"A_Star: Exception: {successor_node}\n\t{e}")
                    exit()
        time_end = datetime.now()
        self.time_execute = time_end - time_start
        logger.info(f"Executed A_Star.\n\tPath found: False\n\tElapsed time: {self.time_execute}\n")
        # return none if goal was not reached and open list is empty
        return None
