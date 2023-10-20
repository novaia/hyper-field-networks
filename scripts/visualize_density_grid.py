from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz import Spherecloud, Lines
from simple_3dviz import Mesh
from simple_3dviz.utils import render
import numpy as np
import argparse

def create_point_cloud(densities, density_threshold):
    x_coordinates, y_coordinates, z_coordinates = np.meshgrid(
        np.linspace(0, 1, densities.shape[0]),
        np.linspace(0, 1, densities.shape[1]),
        np.linspace(0, 1, densities.shape[2])
    )
    coordinates = np.stack([
        np.ravel(x_coordinates), np.ravel(y_coordinates), np.ravel(z_coordinates)
    ], axis=-1)

    flattened_densities = np.ravel(densities)
    thresholded_indices = np.where(flattened_densities > density_threshold)
    thresholded_coordinates = coordinates[thresholded_indices]

    point_cloud = Spherecloud(
        centers=thresholded_coordinates-0.5,
        colors=(0.5, 0.5, 0.5),
        sizes=0.01
    )
    return point_cloud

def create_voxel_grid_mesh(densities, density_threshold):
    voxel_densities = np.squeeze(densities)
    occupied = np.ones(voxel_densities.shape, dtype=bool)
    not_occupied = np.zeros(voxel_densities.shape, dtype=bool)
    voxel_grid = np.where(voxel_densities > density_threshold, occupied, not_occupied)
    voxel_grid_mesh = Mesh.from_voxel_grid(voxel_grid, colors=(0.5, 0.5, 0.5))
    return voxel_grid_mesh

def create_bounding_cube_lines(bound):
    bounding_cube_vertices = np.array([
        [-bound, -bound, -bound], 
        [-bound, -bound,  bound], 
        [-bound,  bound, -bound], 
        [-bound,  bound,  bound], 
        [ bound, -bound, -bound], 
        [ bound, -bound,  bound], 
        [ bound,  bound, -bound], 
        [ bound,  bound,  bound] 
    ])
    bounding_cube_indices = np.array([
        [0, 1], 
        [0, 2], 
        [0, 4], 
        [7, 6], 
        [7, 5], 
        [7, 3], 
        [2, 3], 
        [2, 6], 
        [4, 5], 
        [4, 6], 
        [1, 3], 
        [1, 5] 
    ])
    bounding_cube_points = bounding_cube_vertices[np.ravel(bounding_cube_indices)]
    bounding_cube_lines = Lines(bounding_cube_points, colors=(0.8, 0, 0), width=0.01)
    return bounding_cube_lines

parser = argparse.ArgumentParser()
parser.add_argument('--density_threshold', type=float, default=0.1)
parser.add_argument('--points', type=bool, default=False)
parser.add_argument('--gui', type=bool, default=False)
args = parser.parse_args()

densities = np.load('data/density_grid.npy')

renderables = [create_bounding_cube_lines(0.5)]
if args.points:
    point_cloud = create_point_cloud(densities, args.density_threshold)
    renderables.append(point_cloud)
else:
    voxel_grid_mesh = create_voxel_grid_mesh(densities, args.density_threshold)
    renderables.append(voxel_grid_mesh)

if args.gui:
    from simple_3dviz.window import show
    show(renderables, size=(800, 800), behaviours=[LightToCamera()])
else:
    image = render(
        renderables,
        behaviours=[
            LightToCamera(), 
            SaveFrames("data/density_grid_render_{:03d}.png", every_n=1)
        ],
        n_frames=1,
        camera_position=(-1, -1, -1),
        light=(-60., -160., 120)
    )