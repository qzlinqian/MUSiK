#!/usr/bin/env python
# coding: utf-8

import sys
import os

import numpy as np

import scipy
import tqdm
import open3d as o3d
import matplotlib.pyplot as plt
import glob

import eva_image_quality

parent = os.path.dirname(os.path.realpath('../'))
sys.path.append(parent)

from core import *
from utils import phantom_builder
from utils import geometry

class PoseEvaluator:
    def __init__(self): 
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"]="0"

        self.voxel_size = np.array([0.001, 0.001, 0.001])
        self.surface_mesh = o3d.io.read_triangle_mesh(f"{parent}/assets/kidney_phantom/08_VH_M_renal_papilla_L_i.obj")
        self.body_mask = phantom_builder.voxelize(self.voxel_size[0], mesh=self.surface_mesh)
        self.output_path = f'{parent}/experiment_files/kidney_experiment_opt8_9'
        self.ite_run = 0
        
        self.test_phantom = phantom.Phantom(source_path = None,
                                       voxel_dims = (self.voxel_size[0], self.voxel_size[0], self.voxel_size[0]),
                                       matrix_dims = self.body_mask.shape,
                                       baseline = (1540, 1000),
                                       seed = 5678,
                                       )
        
        blood = tissue.Tissue(name='blood', c=1578, rho=1060, sigma=1.3, scale=0.00001, label=1)
        medulla = tissue.Tissue(name='renal_medulla', c=1564, rho=1044, sigma=40, scale=0.0001, label=2)
        cortex = tissue.Tissue(name='renal_cortex', c=1571.3, rho=1049, sigma=5, scale=0.0001, label=3)
        fat = tissue.Tissue(name='fat', c=1450, rho=920, sigma=0, scale=1, label=4)
        connective = tissue.Tissue(name='connective_tissue', c=1450, rho=1027, sigma=30, scale=0.0001, label=5)
        muscle = tissue.Tissue(name='muscle', c=1580, rho=1090, sigma=0, scale=0.0001, label=6)
        bone = tissue.Tissue(name='bone', c=2500, rho=1800, sigma=0, scale=0.0001, label=7)
        # skin = tissue.Tissue(name='skin', c=1624, rho=1109, sigma=1.3, scale=0.00001, label=8)
        fat = tissue.Tissue(name='fat', c=1450, rho=920, sigma=0, scale=1, label=9)
        
        self.kidney_file_dir = f"{parent}/assets/kidney_phantom/"
        self.kidney_tissue_list = [fat, muscle, muscle, bone, connective, cortex, blood, blood, medulla]
        
        # test_phantom.add_tissue(fat, mask=np.ones(test_phantom.matrix_dims))
        self.test_phantom.build_organ_from_mesh(self.surface_mesh, self.voxel_size[0], self.kidney_tissue_list, dir_path = self.kidney_file_dir)
        self.test_phantom.set_default_tissue('water')
        
        self.simprops = simulation.SimProperties(
                            grid_size   = (150e-3,80e-3,80e-3),
                            voxel_size  = (1e-3,1e-3,1e-3),
                            PML_size    = (8,8,8),
                            PML_alpha   = 2,
                            t_end       = 12e-5,           # [s]
                            bona        = 6,               # parameter b/a determining degree of nonlinear acoustic effects
                            alpha_coeff = 0.5, 	           # [dB/(MHz^y cm)]
                            alpha_power = 1.5,
                            )

    def dummy_run_pose(self, pose_list):
        return np.random.rand(100,100,1)
    
    def run_pose(self, pose_list):
        num_transducers = len(pose_list)
        transducers = [transducer.Planewave(max_frequency=1e6,
                                            elements = 16, 
                                            width = 40e-3,
                                            height =  20e-3,
                                            sensor_sampling_scheme = 'not_centroid', 
                                            sweep = np.pi/6,
                                            ray_num = 8, 
                                            imaging_ndims = 2,
                                            focus_elevation = 20e-3,
                                            ) for i in range(num_transducers)]
        for t in transducers:
            t.make_sensor_coords(1540) # test_phantom.baseline[0]
        
        test_transducer_set = transducer_set.TransducerSet(transducers, seed=8888)
        # pt, normal = test_transducer_set.place_on_mesh_voxel(0, self.surface_mesh,[240,70,330], self.voxel_size[0])
        # pt = np.array([pt[0], pt[1], pt[2]])
        # normal = -np.array([normal[0] - 0.1, normal[1] - 0.5, normal[2] + 0.3])
        theta = np.pi/2
        for i in range(num_transducers):
            pose_diff = pose_list[i]
            pt, normal = test_transducer_set.place_on_mesh_voxel(0, self.surface_mesh, pose_diff[0], self.voxel_size[0])
            pt = np.array([pt[0], pt[1], pt[2]])
            normal = -np.array([normal[0], normal[1], normal[2]]) + pose_diff[1]
            # pt0 = pt + pose_diff[0]
            # normal0 = normal + pose_diff[1]
            about_nl_axis = geometry.Transform(rotation=tuple(theta*normal), translation=(0, 0, 0), about_axis=True)
            pose = geometry.Transform.make_from_heading_vector(normal, pt)
            transducer_pose = about_nl_axis * pose
            test_transducer_set.assign_pose(i, transducer_pose)
            
        test_sensor = sensor.Sensor(transducer_set=test_transducer_set, aperture_type='transmit_as_receive')
        sensor_coord = np.mean(test_sensor.sensor_coords, axis=0) / self.voxel_size + np.array(self.test_phantom.matrix_dims)/2
    
        test_experiment = experiment.Experiment(
                         simulation_path = f'{self.output_path}/{self.ite_run}/',
                         sim_properties  = self.simprops,
                         phantom         = self.test_phantom,
                         transducer_set  = test_transducer_set,
                         sensor          = test_sensor,
                         nodes           = 1,
                         results         = None,
                         indices         = None,
                         workers         = 3,
                         additional_keys = []
                         )
        test_experiment.save()
        test_experiment = experiment.Experiment.load(f'{self.output_path}/{self.ite_run}/')
        test_experiment.run(dry=True)
        test_experiment.run(repeat=True)
        test_experiment.add_results()
        
        test_reconstruction = reconstruction.Compounding(experiment=test_experiment)
        image_matrices = test_reconstruction.compound(workers=16, resolution_multiplier=4, return_local=True)
        np.savez(f'{self.output_path}/{self.ite_run}/image.npz', image=image_matrices, pose=pose_list)
        self.ite_run = self.ite_run+1
        print("Saved image and pose to file")
        return image_matrices

