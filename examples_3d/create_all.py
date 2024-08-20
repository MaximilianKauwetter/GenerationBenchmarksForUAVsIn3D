from commonsky.scenario.scenario import ScenarioID

from examples_3d.helper.create_scenarios import *
from examples_3d.helper.file_helper import create_yaml, write_2_file
import logging

"""
File which creates all_scenarios xml files and configures a default yaml configuration file
RANDOM SEED for replication purpose
"""

RANDOM_SEED = 42

logger = logging.Logger(name="CreateExampleScenarios")
logger.addHandler(logging.StreamHandler())

ALL_SCENARIOS = {
    "1_DirectSimpleTrajectory": create_direct_simple_trajectory,
    "2_SingleStaticObstacle": create_single_static_obstacle,
    "3_ThroughTheForrest": create_through_the_forrest,
    "4_GoingUpThePowerLine": create_going_up_the_powerline,
    "5_InspectingPowerLine": create_inspecting_powerline,
    "6_SplittingPowerLine": create_splitting_powerline,
    "7_FlyingBackPowerline": create_flying_back_powerline,
    "8_SingleDynamicObstacle": create_single_dynamic_obstacle,
    "9_FlyingThroughLinearBirdSwarm": create_flying_trough_linear_bird_swarm,
    "10_FlyingThroughTurningBirdSwarm": create_flying_trough_turning_bird_swarm,
}

if __name__ == "__main__":
    DICT_PATH = dirname(dirname(abspath(__file__)))
    for scenario_file, func in ALL_SCENARIOS.items():
        map_id, map_name = scenario_file.split("_", 1)
        map_id = int(map_id)
        logger.info(f"Create {map_name}(map_id={map_id})")

        filepath_xml = f"{DICT_PATH}\\examples_3d\\scenarios_3d\\{scenario_file}.xml"
        filepath_yaml = f"{DICT_PATH}\\configurations_3d\\{scenario_file}.yaml"

        create_yaml(filepath_yaml)

        test_scenario = Scenario(
            dt=1.0,
            scenario_id=ScenarioID(
                cooperative=False,
                country_id="DEU",
                map_name=map_name,
                map_id=map_id,
            ),
        )
        test_planning_problem_set = PlanningProblemSet()

        func(
            scenario=test_scenario,
            planning_problem_set=test_planning_problem_set,
            random_seed=RANDOM_SEED,
        )

        write_2_file(
            filename=filepath_xml,
            scenario=test_scenario,
            planning_problem_set=test_planning_problem_set,
        )
