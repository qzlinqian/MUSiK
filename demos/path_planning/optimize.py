#!/usr/bin/env python
# coding: utf-8

import evaluate_pose
import eva_image_quality
import numpy as np
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination
import json
# import os

def result_to_json_serializable(result):
    # Extract the necessary data
    return {
        "X": result.X.tolist(),  # Convert numpy arrays to lists
        "F": result.F.tolist(),  # Convert objective function values to lists
        "G": result.G.tolist() if result.G is not None else None,  # Constraints
        "CV": result.CV.tolist() if result.CV is not None else None,  # Constraint violation
        "pop": result.pop.tolist() if result.pop is not None else None,
        "history": result.history.tolist() if result.history is not None else None,
        "exec_time": str(result.exec_time),
        "algorithm": str(result.algorithm)  # Algorithm metadata
    }

def save_results_as_json(result, file_path):
    json_data = result_to_json_serializable(result)
    with open(file_path, "w") as f:
        json.dump(json_data, f, indent=4)

class UltrasoundOptimization(Problem):
    def __init__(self):
        super().__init__(n_var=6,  # Number of scanning parameters
                         n_obj=4,  # Number of objectives (single or multi-objective)
                         n_constr=0,  # Number of constraints
                         xl=[-5, -5, -5, -0.5, -0.5, -0.5],  # Lower bounds for parameters
                         xu=[60, 70, 115, 0.5, 0.5, 0.5])  # Upper bounds for parameters

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = [objective_function(params) for params in x]


np.random.seed()
pose_eva = evaluate_pose.PoseEvaluator()

def objective_function(parameters):
    # Simulate ultrasound image using the input scanning parameters
    pose = [[parameters[0], parameters[1], parameters[2]], [parameters[3], parameters[4], parameters[5]]]
    pose_list = [pose]
    simulated_image = pose_eva.run_pose(pose_list)
    
    # Evaluate the image quality using a custom quality metric
    quality_metric = eva_image_quality.evaluate_image_quality(simulated_image.squeeze())
    
    return -np.array(quality_metric)  # Assuming maximization of quality, use negative for minimization

algorithm = NSGA2(pop_size=5, save_history=True)  # Adjust population size as needed
termination = get_termination("n_gen", 20)  # Run for 100 generations

res = minimize(UltrasoundOptimization(),
               algorithm,
               termination,
               seed=1,  # For reproducibility
               save_history=True,
               verbose=True)

# print(dir(res))
# job_id = os.getenv("SLURM_JOB_ID", "test_job")  # Get SLURM job ID or use a fallback
save_results_as_json(res, "results.json")

# normal = np.array([0, 0, 0])
# pt = np.array([0, 0, 0])
# # pose = geometry.Transform.make_from_heading_vector(normal, pt)
# pose = [pt, normal]
# max_ite = 50
# for i in range(max_ite):
#     pt = (np.random.rand(3)-0.5) * 0.03
#     normal = (np.random.rand(3)-0.5) * 0.03
#     pose = [pt, normal]
#     print(f"Start simulation at new pose {pose}")
#     pose_list = [pose]
#     images = pose_eva.run_pose(pose_list, i+70)
#     matrics = pose_eva.evaluate_images(images)
#     print(matrics)

    # do optimization

    # change pose for next ite
    # pt = (np.random.rand(3,1)-0.5) * 0.03
    # normal = np.random.rand(3,1)-0.5
    # pose = [pt, normal]
