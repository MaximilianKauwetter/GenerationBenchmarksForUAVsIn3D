import numpy as np
from commonsky.geometry.shape import Shape, ShapeGroup, Mesh
from trimesh.collision import CollisionManager
from trimesh import PointCloud

from commonsky.scenario.obstacle import (
    Obstacle,
    StaticObstacle,
    EnvironmentObstacle,
    DynamicObstacle,
    PhantomObstacle,
)


class CollisionChecker3D:
    """
    Collision manager class to detect collisions at any timestep between obstacles and a UAV
    """

    def __init__(self, uav_shape: Mesh, obstacles: None | list[Obstacle] = None):
        self._dynamic_obstacles = []
        self._static_obstacles = []
        self._static_collision_manager = CollisionManager()
        if not isinstance(uav_shape, Mesh):
            raise TypeError()
        self.uav_shape: Mesh = uav_shape
        self.uav_radius = CollisionChecker3D.calc_radius(uav_shape)
        if obstacles is not None:
            self.add_obstacles(obstacles)

    @property
    def uav_shape(self) -> Mesh:
        return self._uav_shape

    @uav_shape.setter
    def uav_shape(self, value: Mesh) -> None:
        self._uav_shape: Mesh = value

    @property
    def uav_radius(self) -> float:
        return self._uav_radius

    @uav_radius.setter
    def uav_radius(self, value: float) -> None:
        self._uav_radius: float = value

    @property
    def static_obstacles(self) -> list[Obstacle]:
        return self._static_obstacles

    @static_obstacles.setter
    def static_obstacles(self, value: list) -> None:
        self._static_obstacles: list = value

    @property
    def dynamic_obstacles(self) -> list[tuple[Obstacle, float]]:
        return self._dynamic_obstacles

    @dynamic_obstacles.setter
    def dynamic_obstacles(self, value: list[tuple[Obstacle, float]]) -> None:
        self._dynamic_obstacles: list[tuple[Obstacle, float]] = value

    def add_obstacles(self, obstacles: list[Obstacle]):
        """
        :param obstacles: list of obstacles
        :return: None

        Adds multiple obstacle to the collision manager
        """
        for obstacle in obstacles:
            self.add_obstacle(obstacle)

    def add_obstacle(self, obstacle: Obstacle):
        """
        :param obstacle: Obstacle that is added to the collision manager
        :return: None

        Adds an obstacle to the collision manager
        """
        shape = obstacle.obstacle_shape
        if isinstance(obstacle, (StaticObstacle, EnvironmentObstacle)):
            self.static_obstacles.append(obstacle)
            if isinstance(shape, ShapeGroup):
                for shape_group_shape in shape.shapes:
                    if not isinstance(shape_group_shape, Mesh):
                        raise TypeError()
                    shape_trimesh = shape_group_shape.tri_mesh
                    if shape_trimesh is None:
                        mesh = shape_group_shape.apply_attributes()
                        shape_trimesh = PointCloud(mesh.reshape(3 * mesh.shape[0], 3)).convex_hull
                    self._static_collision_manager.add_object("mesh", shape_trimesh)
            elif isinstance(shape, Mesh):
                shape_trimesh = shape.tri_mesh
                if shape_trimesh is None:
                    mesh = shape.apply_attributes()
                    shape_trimesh = PointCloud(mesh.reshape(3 * mesh.shape[0], 3)).convex_hull
                self._static_collision_manager.add_object("mesh", shape_trimesh)
            else:
                raise TypeError()
        elif isinstance(obstacle, (DynamicObstacle, PhantomObstacle)):
            radius = CollisionChecker3D.calc_radius(obstacle.obstacle_shape)
            self.dynamic_obstacles.append((obstacle, radius))
        else:
            raise TypeError()

    def check_collision(self, position: np.ndarray, timestep: int) -> bool:
        """
        :param position: position of  the UAV at the timestep
        :param timestep: timestep at which for collisions is checked
        :return: bool if any collision between the UAV and any obstacle is detected

        Checks for any collisions between the UAV and obstacles
        """

        self.uav_shape.center = position
        mesh = self.uav_shape.apply_attributes()
        mesh_reshaped = mesh.reshape(3 * mesh.shape[0], 3)
        tri_mesh = PointCloud(mesh_reshaped).convex_hull

        # static obstacles
        if self._static_collision_manager.in_collision_single(tri_mesh):
            return True

        # dynamic obstacles
        dcm = CollisionManager()
        for ob, ob_radius in self.dynamic_obstacles:
            occ = ob.occupancy_at_time(timestep)
            if occ is None:
                continue
            ob_shape = occ.shape
            if self.uav_radius + ob_radius < np.linalg.norm(ob_shape.center - position):
                continue
            ob_shapes = []
            if isinstance(ob_shape, ShapeGroup):
                ob_shapes = ob_shape.shapes
            else:
                ob_shapes.append(ob_shape)
            for shape in ob_shapes:
                if not isinstance(shape, Mesh):
                    raise TypeError()
                mesh = shape.apply_attributes()
                dcm.add_object("mesh", PointCloud(mesh.reshape(3 * mesh.shape[0], 3)).convex_hull)
        return dcm.in_collision_single(tri_mesh)

    @staticmethod
    def calc_radius(shape: Shape) -> float:
        """
        :param shape: shape that is approximated
        :return: float maximum distance between center and vertices of mesh

        Calculates the maximum radius of a shape as approximation for a convex hull
        """
        if isinstance(shape, ShapeGroup):
            vertices = []
            for s in shape.shapes:
                if not isinstance(s, Mesh):
                    raise TypeError()
                mesh = s.apply_attributes()
                vertices.extend(mesh.reshape(3 * mesh.shape[0], 3))
            vertices = np.array(vertices) - shape.center
            return max([np.linalg.norm(vector) for vector in vertices])
        elif isinstance(shape, Mesh):
            mesh = shape.apply_attributes()
            vertices = mesh.reshape(3 * mesh.shape[0], 3) - shape.center
            return max([np.linalg.norm(vector) for vector in vertices])
        raise TypeError()
