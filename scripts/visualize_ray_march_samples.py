from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.io import SaveFrames
from simple_3dviz import Spherecloud, Lines
from simple_3dviz import Mesh
from simple_3dviz.utils import render
import numpy as np
import argparse

def create_point_cloud(points):
    point_cloud = Spherecloud(
        centers=points,
        colors=(0.5, 0.5, 0.5),
        sizes=0.01
    )
    return point_cloud

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
parser.add_argument('--gui', type=bool, default=False)
args = parser.parse_args()

positions = np.load('data/ray_marched_positions.npy')
origins = np.load('data/ray_origins.npy')

sample_cloud = Spherecloud(centers=positions, colors=(0.5, 0.5, 0.5), sizes=0.01)
origin_cloud = Spherecloud(centers=origins, colors=(0, 0, 0.8), sizes=0.02)
bounding_cube_lines = create_bounding_cube_lines(1)
renderables = [sample_cloud, origin_cloud, bounding_cube_lines]

if args.gui:
    from simple_3dviz.window import show
    show(renderables, size=(800, 800), behaviours=[LightToCamera()])
else:
    image = render(
        renderables,
        behaviours=[
            LightToCamera(), 
            SaveFrames("data/ray_march_samples_render_{:03d}.png", every_n=1)
        ],
        n_frames=1,
        camera_position=(-1, -1, -1),
        light=(-60., -160., 120)
    )