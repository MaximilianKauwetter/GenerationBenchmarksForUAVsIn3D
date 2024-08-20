# CommonSky Motion Planning Library

## Introduction
As part of Maximilian Kauwetter's Bachelor Thesis, the CommonSky Motion Planning Library (MPL) integrates the A* and RRT* for benchmarking UAV motion planning in a complex 3-dimensional environment. 

## Requirements
The required dependencies for running the CommonSky Motion Planner are:

* commonskyio
* omegaconf
* PyYAML
* numpy
* sympy
* trimesh
* rtree
* setuptools
* ffmpeg

python version = 3.10

## Execution

Create all scenario and yaml files:
* Run "examples_3d/create_all.py"

Benchmark execution:
* Run "examples_3d/execute_all.py" 
* Logging file : "example_3d_execute_all_log", where benchmarks are appended

Example Scenario with visualization:
* Run "examples_3d/example_3d"
    

