import logging
import time
import numpy as np
import multiprocessing
from datetime import datetime, timedelta
from os.path import dirname, abspath

from motion_planner_3d.motion_planner_3d import MotionPlanner3D
from motion_planner_config_3d.configuration_3d import Configuration3D
from motion_planner_config_3d.configuration_builder_3d import ConfigurationBuilder3D

from examples_3d.helper.file_helper import create_yaml

"""
Executes the 10 scenarios each with 8 different trajectory planner configuration and benchmark these.
Each benchmark consists of 55 runs where the average is taken.
If after 10 minutes nothing is returned the run is terminated.
"""

# --------------------- constants ---------------------
RANDOM_SEED: int = 42
REPETITIONS: int = 5
TERMINATE_AFTER_SECONDS = 600
LOGGING_LEVEL = logging.NOTSET
PATH_COMMONSKY = dirname(dirname(abspath(__file__)))
ConfigurationBuilder3D.set_path_to_config(f"{PATH_COMMONSKY}\\configurations_3d")


logger = logging.Logger(name="EXECUTE_ALL_LOGGER")
logger.setLevel(LOGGING_LEVEL)
logger.addHandler(logging.StreamHandler())

csv_logger = logging.Logger("CSV_LOGGER")
csv_logger.setLevel(logging.NOTSET)
csv_logger.addHandler(logging.FileHandler(filename="execute_all_log.csv"))


# --------------------- functions ---------------------
def array2log(array, seperator: str) -> str:
    """
    :param array: array containing the raw log
    :param seperator: seperator of the log file
    :return: formatted log

    formats a string array to a formatted log str array
    """

    ret = []
    for i in array:
        if isinstance(i, timedelta):
            ret.append(str(i).replace(".", ":"))
        else:
            ret.append(str(i))
    return seperator.join(ret)


def run_planning(config: Configuration3D, execute_benchmarks):
    """
    :param config: configuration for the motion planner
    :param execute_benchmarks: file to put the results
    :return: None

    Runs a trajectory planning and stores the benchmark
    """
    # execute
    mp = MotionPlanner3D(config)
    mp.plan()

    # analyse results
    path_found = mp.planned_trajectory is not None
    if path_found:
        number_trajectory_states = len(mp.planned_trajectory.state_list)
        traveled_distance = sum(
            [np.linalg.norm(state.velocity) for state in mp.planned_trajectory.state_list]
        )
        total_costs = (
            getattr(mp.trajectory_planner, "costs")
            if hasattr(mp.trajectory_planner, "costs")
            else 0
        )
        final_time_step = mp.planned_trajectory.final_state.time_step
    else:
        number_trajectory_states = 0
        traveled_distance = 0
        total_costs = 0
        final_time_step = 0

    time_init = (
        getattr(mp.trajectory_planner, "time_init")
        if hasattr(mp.trajectory_planner, "time_init")
        else 0
    )
    time_execute = (
        getattr(mp.trajectory_planner, "time_execute")
        if hasattr(mp.trajectory_planner, "time_execute")
        else 0
    )
    time_total = time_init + time_execute

    run_benchmark = [
        path_found,
        number_trajectory_states,
        traveled_distance,
        total_costs,
        final_time_step,
        time_init,
        time_execute,
        time_total,
    ]
    execute_benchmarks.put(run_benchmark)


def execute(filename: str, yaml_dict: dict[str, dict[str,]]):
    """
    :param filename: filename of the scenario
    :param yaml_dict: configuration as dict
    :return: None

    Executes a benchmark for a scenario in combination with a trajectory planner configuration
    """
    filepath_yaml = f"{PATH_COMMONSKY}\\configurations_3d\\{filename}.yaml"

    create_yaml(file=filepath_yaml, yaml_dict=yaml_dict)
    time.sleep(5)
    config = ConfigurationBuilder3D.build_configuration(
        name_scenario=filename,
        idx_planning_problem=1,
    )

    csv_logger.info(
        "Run nr.,Path found,Number Trajectory states,Traveled Distance,Total costs,Final time step,Time init,Time execute,Time total"
    )
    execute_benchmarks = []
    for run_nr in range(1, REPETITIONS + 1):
        queue = multiprocessing.Queue()
        run = multiprocessing.Process(target=run_planning, args=(config, queue))
        run.start()
        for sec in range(1, TERMINATE_AFTER_SECONDS + 1):
            if sec % 10 == 0:
                logger.info(f"Sec: {sec}")
            time.sleep(1)
            if not run.is_alive():
                run_benchmark = queue.get()
                break
        else:
            run.terminate()
            run.join()
            run_benchmark = [False, 0] + [0] * 6
            logger.warning("Run terminated")
        while not queue.empty():
            queue.get()

        execute_benchmarks.append(run_benchmark)
        csv_logger.info(f"{run_nr},{array2log(run_benchmark,',')}")

    avg = np.average(execute_benchmarks, axis=0)
    csv_logger.info(f"AVG,{array2log(avg,',')}\n")


# --------------------- properties  ---------------------
PROPERTIES_SIMPLE_FILE = dict(
    grid_grid_size=0.5,
    number_new_states_per_rad=37,
    total_states=1000,
    bounding_box="all",
)

PROPERTIES_SPECIAL_FILE = dict(
    grid_grid_size=0.5,
    number_new_states_per_rad=37,
    total_states=50000,
    bounding_box="all",
)

PROPERTIES_APPLICATION_FILE = dict(
    grid_grid_size=0.5,
    number_new_states_per_rad=37,
    total_states=50000,
    bounding_box="planning_problem",
)

# --------------------- file names ---------------------
ALL_FILES: dict[str, dict[str,]] = {
    "1_DirectSimpleTrajectory": PROPERTIES_SIMPLE_FILE,
    "2_SingleStaticObstacle": PROPERTIES_SIMPLE_FILE,
    "3_ThroughTheForrest": PROPERTIES_SPECIAL_FILE,
    "4_GoingUpThePowerLine": PROPERTIES_APPLICATION_FILE,
    "5_InspectingPowerLine": PROPERTIES_APPLICATION_FILE,
    "6_SplittingPowerLine": PROPERTIES_APPLICATION_FILE,
    "7_FlyingBackPowerline": PROPERTIES_APPLICATION_FILE,
    "8_SingleDynamicObstacle": PROPERTIES_SIMPLE_FILE,
    "9_FlyingThroughLinearBirdSwarm": PROPERTIES_APPLICATION_FILE,
    "10_FlyingThroughTurningBirdSwarm": PROPERTIES_APPLICATION_FILE,
}

TEST_FILES: dict[str, dict[str,]] = {
    "8_SingleDynamicObstacle": PROPERTIES_SIMPLE_FILE,
    "9_FlyingThroughLinearBirdSwarm": PROPERTIES_APPLICATION_FILE,
}


yaml_basic: dict[str, dict[str,]] = dict(
    planning=dict(
        dt=1,
    ),
    uav=dict(
        uav_id=1,
        radius=1,
        v_min=0,
        v_max=3,
        a_min=-0.5,
        a_max=0.5,
    ),
)


if __name__ == "__main__":
    time_start = datetime.now()
    csv_logger.info(f"RUN: {datetime.now()}")
    csv_logger.info("UAV:")
    uav: dict[str,] = yaml_basic["uav"]
    csv_logger.info(",".join(uav.keys()))
    csv_logger.info(",".join([str(v) for v in uav.values()]))
    csv_logger.info("")

    for file, properties in TEST_FILES.items():
        logger.info(f"################################ {file} ################################")
        csv_logger.info(f"{file}" + ",----------------" * 7)
        # A Star Grid
        logger.info(
            f"-------------------------------- A_Star_Grid -------------------------------- "
        )
        yaml = yaml_basic
        yaml["planning"]["trajectory_planner"] = "a_star"
        yaml["planning"]["motion_primitive"] = "grid"
        yaml["planning"]["grid_size"] = properties["grid_grid_size"]
        for movement_cost_and_heuristic in ["euc_distance", "velocity", "max_velocity"]:
            logger.info(f"A Star Grid: {movement_cost_and_heuristic}")
            yaml["planning"]["movement_cost_and_heuristic"] = movement_cost_and_heuristic
            csv_logger.info("A Star Grid,grid_size,movement_cost_and_heuristic")
            csv_logger.info(f',{properties["grid_grid_size"]},{movement_cost_and_heuristic}')
            execute(filename=file, yaml_dict=yaml)

        # A Star Fibonacci
        logger.info(
            f"-------------------------------- A_Star_Fib -------------------------------- "
        )
        yaml = yaml_basic
        yaml["planning"]["trajectory_planner"] = "a_star"
        yaml["planning"]["motion_primitive"] = "fibonacci_sphere"
        yaml["planning"]["grid_size"] = 0.25
        yaml["planning"]["number_new_states_per_rad"] = properties["number_new_states_per_rad"]
        for movement_cost_and_heuristic in ["euc_distance", "velocity", "max_velocity"]:
            logger.info(f"A Star Fibonacci: {movement_cost_and_heuristic}")
            yaml["planning"]["movement_cost_and_heuristic"] = movement_cost_and_heuristic
            csv_logger.info(
                "A Star Fibonacci,grid_size,number_new_states_per_rad,movement_cost_and_heuristic"
            )
            csv_logger.info(
                f',{yaml["planning"]["grid_size"]},{properties["number_new_states_per_rad"]},{movement_cost_and_heuristic}'
            )
            execute(filename=file, yaml_dict=yaml)

        # RRT Star
        logger.info(f"-------------------------------- RRT_Star -------------------------------- ")
        yaml = yaml_basic
        yaml["planning"]["trajectory_planner"] = "rrt_star"
        yaml["planning"]["grid_size"] = 0.1
        yaml["planning"]["total_states"] = properties["total_states"]
        yaml["planning"]["bounding_box"] = properties["bounding_box"]
        for movement_cost_and_heuristic in ["euc_distance", "velocity"]:
            logger.info(f"RRT Star: {movement_cost_and_heuristic}")
            yaml["planning"]["movement_cost_and_heuristic"] = movement_cost_and_heuristic
            csv_logger.info(
                "RRT Star,grid_size,total_states,bounding_box,movement_cost_and_heuristic"
            )
            csv_logger.info(
                f',{yaml["planning"]["grid_size"]},{properties["total_states"]},{properties["bounding_box"]},{movement_cost_and_heuristic}'
            )
            execute(filename=file, yaml_dict=yaml)

        logger.info(f"Total time passed: {datetime.now()-time_start}")
