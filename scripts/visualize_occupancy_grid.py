from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz import Spherecloud
from simple_3dviz.utils import render
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--density_threshold', type=float, default=6.0)
parser.add_argument('--gui', type=bool, default=False)
args = parser.parse_args()

occupancy_grid_density = np.load('data/occupancy_grid_density.npy')
occupancy_grid_coordinates = np.load('data/occupancy_grid_coordinates.npy')

num_coordinates = occupancy_grid_coordinates.shape[0]
num_densities = occupancy_grid_density.shape[0]
assert num_coordinates == num_densities, 'Number of coordinates and densities must match, ' + \
    f'got {num_coordinates} and {num_densities}'


thresholded_indices = np.where(occupancy_grid_density > args.density_threshold)
thresholded_coordinates = occupancy_grid_coordinates[thresholded_indices]
assert thresholded_coordinates.shape[0] > 0, 'There were no densities above the threshold.'

point_cloud = Spherecloud(
    centers=thresholded_coordinates,
    colors=(0.5, 0.5, 0.5),
    sizes=0.01
)

if args.gui:
    from simple_3dviz.window import show
    show(point_cloud, size=(800, 800), behaviours=[LightToCamera()])
else:
    image = render(
        point_cloud,
        behaviours=[
            LightToCamera(), 
            SaveFrames("data/occupancy_render/frame_{:03d}.png", every_n=1)
        ],
        n_frames=1,
        camera_position=(-1, -1, -1),
        light=(-60., -160., 120)
    )