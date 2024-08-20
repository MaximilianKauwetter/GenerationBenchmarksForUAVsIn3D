import numpy as np

from .motion_primitive_interface_3d import MotionPrimitiveInterface3D

GOLDEN_RATIO = (1 + np.sqrt(5)) / 2


class FibonacciSphereMotionPrimitive(MotionPrimitiveInterface3D):
    """
    Motion Primitive class for generating successor states using the fibonacci sphere algorithm
    """

    grid_size: float
    max_velocity: float
    min_velocity: float
    max_acceleration: float
    min_acceleration: float
    number_new_states_per_rad: int

    def __init__(self, config):
        super().__init__(config)
        self.grid_size = self.config.config_planning.grid_size
        self.min_velocity = self.config.config_uav.v_min
        self.max_velocity = self.config.config_uav.v_max
        self.min_acceleration = self.config.config_uav.a_min
        self.max_acceleration = self.config.config_uav.a_max
        self.number_new_states_per_rad = self.config.config_planning.number_new_states_per_rad

        # unit fib sphere
        self.unit_points = []
        for i in range(self.number_new_states_per_rad):
            z = -1 + (i * 2) / (
                self.number_new_states_per_rad - 1
            )  # z form -1 to 1 equally divided
            xy_radius = np.sqrt(1 - z**2)  # radius left to be subdivided by x and y
            gai = GOLDEN_RATIO * i
            point = np.array([np.sin(gai) * xy_radius, np.cos(gai) * xy_radius, z])
            self.unit_points.append(point)

    def calculate_successor(
        self,
        position: np.ndarray,
        velocity: float,
        goal_position: None | np.ndarray = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        :param position: current position as 3d coordinates
        :param velocity: absolute velocity
        :param goal_position: goal position as 3d coordinates
        :return: list of successor states as tuple of 3d position and 3d movement

        calculates a fixed number of successor states from a given state by using a fibonacci sphere
        """
        if goal_position is not None and self.goal_in_reach(
            position=position,
            velocity=velocity,
            goal_position=goal_position,
        ):
            return [(goal_position, goal_position - position)]

        new_states = []
        for vel in [
            max(velocity + self.min_acceleration, self.min_velocity),
            velocity,
            min(velocity + self.max_acceleration, self.max_velocity),
        ]:
            if np.isclose(vel, 0):
                new_state = (
                    position,
                    np.array([0, 0, 0]),
                )
                new_states.append(new_state)
            else:
                for point in self.unit_points:
                    velocity_3d = point * vel
                    new_position = position + velocity_3d
                    new_state = (
                        np.round(new_position / self.grid_size) * self.grid_size,
                        velocity_3d,
                    )
                    new_states.append(new_state)

        return new_states

    def goal_in_reach(
        self,
        position: np.ndarray,
        velocity: float,
        goal_position: np.ndarray,
    ) -> bool:
        """
        :param position: current position as 3d coordinates
        :param velocity: absolute velocity
        :param goal_position: goal position as 3d coordinates
        :return: bool if goal is in reach

        clarifies if the goal position can be reached from the current position with the current velocity
        """
        # check if no current velocity due to already on position or at start
        if np.isclose(velocity, 0):
            return np.allclose(position, goal_position)
        goal_vector = goal_position - position

        # check possible distance
        lowest_velocity = max(velocity + self.min_acceleration, self.min_velocity)
        highest_velocity = min(velocity + self.max_acceleration, self.max_velocity)
        if not lowest_velocity <= np.linalg.norm(goal_vector) <= highest_velocity:
            return False

        return True
