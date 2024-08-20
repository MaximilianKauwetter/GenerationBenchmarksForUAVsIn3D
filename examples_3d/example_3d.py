from motion_planner_3d.motion_planner_3d import MotionPlanner3D
from motion_planner_config_3d.configuration_builder_3d import ConfigurationBuilder3D
from commonsky.visualization.o3d_renderer import O3DRenderer
from os.path import dirname, abspath

"""
Plans one file and shows the result afterwards with the O3DRenderer
"""

ALL_FILENAMES = [
    "1_DirectSimpleTrajectory",
    "2_SingleStaticObstacle",
    "3_ThroughTheForrest",
    "4_GoingUpThePowerLine",
    "5_InspectingPowerLine",
    "6_SplittingPowerLine",
    "7_FlyingBackPowerline",
    "8_SingleDynamicObstacle",
    "9_FlyingThroughLinearBirdSwarm",
    "10_FlyingThroughTurningBirdSwarm",
]

PATH_COMMONSKY = dirname(dirname(abspath(__file__)))

ConfigurationBuilder3D.set_path_to_config(f"{PATH_COMMONSKY}\\configurations_3d")
config = ConfigurationBuilder3D.build_configuration("1_DirectSimpleTrajectory", 1)

mp = MotionPlanner3D(config)
mp.plan()

rnd = O3DRenderer()
rnd.draw_params.time_begin = 4
rnd.draw_params.planning_problem.initial_state.state.facecolor = "#0000ff"
rnd.draw_params.environment_obstacle.occupancy.shape.facecolor = "#0000ff"
rnd.draw_params.dynamic_obstacle.history.draw_history = True
rnd.draw_params.dynamic_obstacle.trajectory.draw_trajectory = True
mp.draw(rnd)
rnd.render(show=True)
