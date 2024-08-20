import yaml
from commonsky.common.file_writer import CommonSkyFileWriter, OverwriteExistingFile
from commonsky.planning.planning_problem import PlanningProblemSet
from commonsky.scenario.scenario import (
    Scenario,
    Tag,
    Location,
    Environment,
    Time,
    TimeOfDay,
    Weather,
    Underground,
)


def write_2_file(
    filename: str,
    scenario: Scenario,
    planning_problem_set: PlanningProblemSet,
) -> None:
    """
    :param filename: file path and name
    :param scenario: scenario that will be saved
    :param planning_problem_set:  planning problem that will be saved
    :return: None

    Writes a scenario and planning problem to the specified path
    """
    CommonSkyFileWriter(
        scenario=scenario,
        planning_problem_set=planning_problem_set,
        author="Maximilian Kauwetter",
        affiliation="TUMunich",
        source="handcrafted",
        tags={
            Tag.SIMULATED,
        },
        location=Location(
            geo_name_id=2867714,
            gps_latitude=48.13743,
            gps_longitude=11.57549,
            environment=Environment(
                time=Time(
                    hours=10,
                    minutes=30,
                ),
                time_of_day=TimeOfDay.DAY,
                weather=Weather.UNKNOWN,
                underground=Underground.DIRTY,
            ),
        ),
    ).write_to_file(
        filename=filename,
        overwrite_existing_file=OverwriteExistingFile.ALWAYS,
        check_validity=True,
    )


def create_yaml(file: str, yaml_dict=None):
    """
    :param file: yaml file path
    :param yaml_dict: specific yaml dictionary
    :return: None

    Creates a yaml file for the specified file
    """
    if yaml_dict is None:
        planning = dict(
            dt=1,
            trajectory_planner="rrt_star",  # a_star # rrt_star
            grid_size=0.06,
            motion_primitive="grid",  # fibonacci_sphere
            movement_cost_and_heuristic="velocity",  # euc_distance # max_velocity
            total_states=75000,
            number_new_states_per_rad=23,
            bounding_box="planning_problem",  # "all", "static", "dynamic", "planning_problem"
        )
        uav = dict(
            uav_id=1,
            radius=1,
            v_min=0,
            v_max=2,
            a_min=-0.5,
            a_max=0.5,
        )
        yaml_dict = dict(
            planning=planning,
            uav=uav,
        )
    with open(file, "w") as file:
        yaml.dump(yaml_dict, file)
