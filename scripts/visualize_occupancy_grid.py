from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import RotateModel
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz import Spherecloud
from simple_3dviz.utils import render
import jax.numpy as jnp

density_threshold = 6.0
occupancy_grid_density = jnp.load('data/occupancy_grid_density.npy')
occupancy_grid_coordinates = jnp.load('data/occupancy_grid_coordinates.npy')

thresholded_indices = jnp.where(occupancy_grid_density > density_threshold)
thresholded_coordinates = occupancy_grid_coordinates[thresholded_indices]

point_cloud = Spherecloud(
    centers=thresholded_coordinates,
    colors=(0.5, 0.5, 0.5),
    sizes=0.5
)

#show(mesh, size=(800, 600), behaviours=[RotateModel(), LightToCamera()])
image = render(
    point_cloud,
    behaviours=[
        RotateModel(), 
        LightToCamera(), 
        SaveFrames("data/occupancy_render/frame_{:03d}.png", every_n=1)
    ],
    n_frames=1,
    light=(-60., -160., 120)
)