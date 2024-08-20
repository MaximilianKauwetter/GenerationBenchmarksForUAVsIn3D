from abc import abstractmethod

import numpy as np
from motion_planner_config_3d.configuration_3d import Configuration3D


class MotionPrimitiveInterface3D:
    """
    Motion Primitive base class
    """

    config: Configuration3D

    def __init__(self, config: Configuration3D):
        self.config = config

    @abstractmethod
    def calculate_successor(
        self,
        position,
        velocity,
        goal_position: None | np.ndarray = None,
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        calculates all successor states from a given state

        :param position: current position as 3d coordinates
        :param velocity: absolute velocity
        :param goal_position: goal position as 3d coordinates
        :return: list of successor states as tuple of 3d position and 3d movement
        """

        raise NotImplementedError()
