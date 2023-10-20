from simple_3dviz import Mesh
from simple_3dviz.behaviours.misc import LightToCamera
from simple_3dviz.behaviours.movements import RotateModel
from simple_3dviz.behaviours.io import SaveFrames
#from simple_3dviz.window import show
from simple_3dviz.utils import render
import jax.numpy as jnp

array = jnp.load('data/occupancy_grid_bitfield.npy')
array = list(jnp.reshape(array, (64, 64, 64)))
vertices = []
colors = []
for i in range(64):
    for j in range(64):
        for k in range(64):
            if array[i][j][k] == 1:
                vertices.extend([
                    (i-0.4, j-0.4, k-0.4),
                    (i+0.4, j-0.4, k-0.4),
                    (i+0.4, j+0.4, k-0.4),
                    (i-0.4, j+0.4, k-0.4),
                    (i-0.4, j-0.4, k+0.4),
                    (i+0.4, j-0.4, k+0.4),
                    (i+0.4, j+0.4, k+0.4),
                    (i-0.4, j+0.4, k+0.4)
                ])
                colors.extend([(0.5, 0.5, 0.5)]*8)

mesh = Mesh.from_vertices(vertices, colors=colors)

#show(mesh, size=(800, 600), behaviours=[RotateModel(), LightToCamera()])

image = render(
    mesh,
    behaviours=[
        RotateModel(), 
        LightToCamera(), 
        SaveFrames("/data/occupancy_render/frame_{:03d}.png", every_n=5)
    ],
    n_frames=512,
    camera_position=(-60., -160., 120),
    camera_target=(0., 0., 40),
    light=(-60., -160., 120)
)