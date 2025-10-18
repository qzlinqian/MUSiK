#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import csv

parent = os.path.dirname('/home/qalin/ultrasound/MUSiK/')
sys.path.append(parent)
sys.path.append('/workspace')  # or wherever the project root is inside the container

import numpy as np


import scipy
import tqdm
import open3d as o3d
import matplotlib.pyplot as plt
import glob
from scipy.spatial.transform import Rotation as R

from core import *
from utils import phantom_builder
from utils import geometry

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"


voxel_size = np.array([0.0005, 0.0005, 0.0005])
surface_mesh = o3d.io.read_triangle_mesh(f"{parent}/assets/kidney_phantom/00_abdomen_cropped.obj")
body_mask = phantom_builder.voxelize(voxel_size[0], mesh=surface_mesh)
simulation_dir = f'{parent}/experiment_files/kidney_experiment_02'


test_phantom = phantom.Phantom(source_path = None,
                               voxel_dims = (voxel_size[0], voxel_size[0], voxel_size[0]),
                               matrix_dims = body_mask.shape,
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

kidney_file_dir = f"{parent}/assets/kidney_phantom/"
kidney_tissue_list = [fat, muscle, muscle, bone, connective, cortex, blood, blood, medulla]

# test_phantom.add_tissue(fat, mask=np.ones(test_phantom.matrix_dims))
test_phantom.build_organ_from_mesh(surface_mesh, voxel_size[0], kidney_tissue_list, dir_path = kidney_file_dir)
test_phantom.set_default_tissue('water')


# open best_path for next scan
contact_points = []
rotations = []

with open(f'{parent}/experiment_files/best_path.txt', "r") as f:
    for line in f:
        values = list(map(float, line.strip().split()))
        if len(values) == 7:
            point = np.array([values[0], values[1], values[2]])
            orient = np.array([values[3], values[4], values[5], values[6]])
            rot = R.from_quat(orient).as_rotvec()  # x y z w
            contact_points.append(point)
            rotations.append(-rot)

num_transducers = len(contact_points)
ray_number = 32

transducers = [transducer.Planewave(max_frequency=1e6,
                                    elements = 32, 
                                    width = 40e-3,
                                    height =  20e-3,
                                    sensor_sampling_scheme = 'not_centroid', 
                                    sweep = np.pi/3,
                                    ray_num = ray_number, 
                                    imaging_ndims = 2,
                                    focus_elevation = 20e-3,
                                    ) for i in range(num_transducers)]

for t in transducers:
    t.make_sensor_coords(1540) # test_phantom.baseline[0]

test_transducer_set = transducer_set.TransducerSet(transducers, seed=8888)


# pt, normal = test_transducer_set.place_on_mesh_voxel(0, surface_mesh,[0,-10,-50], voxel_size[0])
# pt = np.array([pt[0], pt[1], pt[2]])
# normal = -np.array([normal[0], normal[1], normal[2]])

# pose = geometry.Transform.make_from_heading_vector(normal, pt)

# theta = np.pi/2
theta = 0
for i in range(num_transducers):
    # about_nl_axis = geometry.Transform(rotation=tuple(theta), translation=(0, 0, 0), about_axis=True)
    pose = geometry.Transform.make_from_heading_vector(rotations[i], contact_points[i])
    
    transducer_pose = pose
    
    test_transducer_set.assign_pose(i, transducer_pose)

test_sensor = sensor.Sensor(transducer_set=test_transducer_set, aperture_type='transmit_as_receive')

sensor_coord = np.mean(test_sensor.sensor_coords, axis=0) / voxel_size + np.array(test_phantom.matrix_dims)/2


# points = np.array((o3d.io.read_triangle_mesh(f"{parent}/assets/kidney_phantom/00_abdomen_cropped.obj")).sample_points_uniformly(1000).points)
points = np.array((surface_mesh).sample_points_uniformly(1000).points)
points = points[:,[0,1,2]] - np.mean(points, axis=0)
test_transducer_set.plot_transducer_coords(scale = 0.2, phantom_coords = points, view=(30,20))
plt.savefig("transducers.png", dpi=300, bbox_inches='tight')  # dpi for resolution, bbox_inches to remove extra whitespace


simprops = simulation.SimProperties(
                grid_size   = (150e-3,80e-3,80e-3),
                voxel_size  = (0.5e-3,0.5e-3,0.5e-3),
                PML_size    = (8,8,8),
                PML_alpha   = 2,
                t_end       = 12e-5,           # [s]
                bona        = 6,               # parameter b/a determining degree of nonlinear acoustic effects
                alpha_coeff = 0.5, 	           # [dB/(MHz^y cm)]
                alpha_power = 1.5,
                )


test_experiment = experiment.Experiment(
                 simulation_path = simulation_dir,
                 sim_properties  = simprops,
                 phantom         = test_phantom,
                 transducer_set  = test_transducer_set,
                 sensor          = test_sensor,
                 nodes           = 1,
                 results         = None,
                 indices         = None,
                 workers         = 3,
                 additional_keys = []
                 )

test_experiment.save()

test_experiment = experiment.Experiment.load(simulation_dir)
test_experiment.run(dry=True)


test_experiment.run(repeat=False)

test_experiment.add_results()

test_reconstruction = reconstruction.Compounding(experiment=test_experiment)

image_matrices = test_reconstruction.compound(combine = False, workers=16, resolution_multiplier=4, return_local=True) ######### ANK: we use the combine=False flag to get back one "self-view" image for each of the transducers. In this case, 10 different images should result.
print("Reconstruction completed. Now combine and save rays.")

#### reconstruction and save
image_matrices = np.array(image_matrices)
image_matrices = np.squeeze(image_matrices)
subfolder = f"{simulation_dir}/output_files"

# Create the subfolder if it doesn't exist
os.makedirs(subfolder, exist_ok=True)

image_names = []
for i in range(num_transducers):
    # reconstruct on transducer at a time
    image = image_matrices[i*ray_number:(i+1)*ray_number, :, :]
    image = np.sum(image, axis=0)
    image = image / np.max(image) * 255
    
    # Save as .npy file
    npy_filename = os.path.join(subfolder, f"image_{i}.npy")
    np.save(npy_filename, image)
    
    # Save as .png image
    image_filename = os.path.join(subfolder, f"image_{i}.png")
    plt.imsave(image_filename, image, cmap='gray')
    image_names.append(image_filename)

    print(f"Saved files: {npy_filename}, {image_filename}")

# Write pose to CSV
# TODO: test pose save!!!!!!
with open("poses.csv", "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "x", "y", "z", "qx", "qy", "qz", "qw"])  # header
    for i, pose in zip(range(num_transducers), test_transducer_set.transmit_poses()):
        T = pose.get()
        x, y, z = T[:3, 3]
        rotation_matrix = T[:3, :3]
        r = R.from_matrix(rotation_matrix)
        qx, qy, qz, qw = r.as_quat()
        writer.writerow([f"image_{i}.png", x, y, z, qx, qy, qz, qw])

# np.save('test_image_0.npy', image_matrices)

# image_matrices = np.array(image_matrices)
# print(image_matrices.shape)

