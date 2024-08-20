import os

from commonsky.common.file_reader import CommonSkyFileReader

from motion_planner_config_3d.configuration_3d import Configuration3D
from omegaconf import OmegaConf, ListConfig, DictConfig


class ConfigurationBuilder3D:
    path_config: str = None
    path_config_default: str = None

    @classmethod
    def set_path_to_config(cls, path_config: str, dir_configs_default: str = "defaults_3d"):
        cls.path_config = path_config
        cls.path_config_default = os.path.join(path_config, dir_configs_default)

    @classmethod
    def build_configuration(
        cls, name_scenario: str, idx_planning_problem: int = 0
    ) -> Configuration3D:
        config_default = cls.construct_default_config()
        config_scenario = cls.construct_scenario_configuration(name_scenario)
        config_command_line_interface = OmegaConf.from_cli()

        config_combined = OmegaConf.merge(
            config_default,
            config_scenario,
            config_command_line_interface,
        )
        config = Configuration3D(config_combined)

        scenario, planning_problem_set = CommonSkyFileReader(config.general.path_scenario).open()
        planning_problem = planning_problem_set.find_planning_problem_by_id(idx_planning_problem)
        config.complete_configuration(
            scenario=scenario,
            planning_problem=planning_problem,
            planning_problem_set=planning_problem_set,
        )

        return config

    @classmethod
    def construct_default_config(cls) -> ListConfig | DictConfig:
        """
        Constructs default configuration by accumulating yaml files.
        Collects all motion_planner_config files ending with .yaml under path_config_default.
        """

        config_default = OmegaConf.create()
        try:
            OmegaConf.register_new_resolver(
                "join_paths",
                lambda base_path, additional_path: os.path.join(base_path, additional_path),
            )
        except ValueError as value_error:
            print("Re-attempting to register join_paths resolver exception is suppressed.")

        for yaml_files in [
            os.path.join(cls.path_config_default, path_file)
            for path_file in os.listdir(cls.path_config_default)
            if path_file.endswith(".yaml")
        ]:
            with open(yaml_files, "r") as file_config:
                try:
                    config_partial = OmegaConf.load(file_config)
                    OmegaConf.resolve(config_partial)
                    name_file = os.path.basename(yaml_files).split(".")[0]

                except Exception as e:
                    print(e)

                else:
                    config_default[name_file] = config_partial

        if config_default.get("general", None) is not None:
            for key, path in config_default["general"].items():
                if not key.startswith("path_"):
                    continue
                path_relative = os.path.join(cls.path_config, path)
                if os.path.exists(path_relative):
                    config_default["general"][key] = path_relative

        return config_default

    @classmethod
    def construct_scenario_configuration(cls, name_scenario: str):
        """Constructs scenario-specific configuration."""
        config_scenario = OmegaConf.create()

        path_config_scenario = f"{os.path.join(cls.path_config,name_scenario)}.yaml"
        if os.path.exists(path_config_scenario):
            with open(path_config_scenario, "r") as file_config:
                try:
                    config_scenario = OmegaConf.load(file_config)

                except Exception as e:
                    print(e)

                else:
                    # add scenario name to the config file
                    config_scenario["general"] = {"name_scenario": name_scenario}

        return config_scenario
