from abc import ABC, abstractmethod
from typing import Dict, List
from enum import Enum
from commonsky.prediction.prediction import Prediction

# from commonsky.visualization.mp_renderer_3d import MPRenderer3D
from commonsky.visualization.o3d_renderer import O3DRenderer
from commonsky.prediction.prediction import TrajectoryPrediction, Occupancy


class PredictorType3D(Enum):
    """Enum containing all possible predictor types defined in MPL.
    MOTION_MODEL: motion model predictor
    """

    MOTION_MODEL = "motion_model"

    @classmethod
    def values(cls):
        return [item.value for item in cls]


class PredictorInterface3D(ABC):
    """
    Base class for prediction.
    """

    @property
    @abstractmethod
    def trajectory_predicted(self) -> Dict[int, TrajectoryPrediction]:
        return self.trajectory_predicted

    @property
    @abstractmethod
    def scenario(self):
        return self.scenario

    def predict(self) -> Dict[int, Prediction]:
        """Abstract method for predictions."""
        pass

    def visualize(self):
        """
        Visualize the prediction
        Check if prediction has been made first
        """
        if not bool(self.trajectory_predicted):
            raise Exception("In order to visualize predict() has to be executed first")

        rnd: O3DRenderer = O3DRenderer()
        self.scenario.draw(rnd)

        for p in self.trajectory_predicted.values():
            occupancy_set: List[Occupancy] = p.occupancy_set
            for o in occupancy_set:
                o.draw(rnd)

        rnd.render(show=True)
