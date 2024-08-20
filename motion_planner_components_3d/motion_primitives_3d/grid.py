import numpy as np
from .motion_primitive_interface_3d import MotionPrimitiveInterface3D

PRECISION = 10


class GridMotionPrimitive(MotionPrimitiveInterface3D):
    """
    Motion Primitive class for generating all possible successor states.
    """

    grid_size: float
    max_velocity: float
    min_velocity: float
    max_acceleration: float
    min_acceleration: float

    def __init__(self, config):
        super().__init__(config)
        self.grid_size = self.config.config_planning.grid_size
        self.min_velocity = self.config.config_uav.v_min
        self.max_velocity = self.config.config_uav.v_max
        self.min_acceleration = self.config.config_uav.a_min
        self.max_acceleration = self.config.config_uav.a_max

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

        calculates all possible successor states from a given state
        """
        lowest_velocity = max(velocity + self.min_acceleration, self.min_velocity)
        highest_velocity = min(velocity + self.max_acceleration, self.max_velocity)

        new_states = []
        grid_points = np.concatenate(
            (
                np.flip(
                    np.arange(
                        start=0,
                        stop=-highest_velocity - self.grid_size,
                        step=-self.grid_size,
                    )
                ),
                np.arange(
                    start=self.grid_size,
                    stop=highest_velocity + self.grid_size,
                    step=self.grid_size,
                ),
            )
        )
        for x in grid_points:
            for y in grid_points:
                for z in grid_points:
                    new_movement = np.array([x, y, z])
                    velocity_norm = np.linalg.norm(new_movement)
                    if lowest_velocity <= velocity_norm <= highest_velocity:
                        new_states.append(
                            (np.round(position + new_movement, PRECISION), new_movement)
                        )
        return new_states
