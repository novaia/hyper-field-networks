from typing import Callable, List, Literal, Tuple, Type, Union
from flax.struct import dataclass
from flax.training.train_state import TrainState
from flax import struct
import jax
import jax.numpy as jnp
import jax.random as jran
from volrendjax import packbits, morton3d_invert
import numpy as np
import dataclasses

@dataclass 
class StateOptions:
    diagonal_n_steps:int
    density_grid_res:int
    stepsize_portion:float
    scene_bound:float
    camera_cx:float
    camera_cy:float
    camera_fx:float
    camera_fy:float
    camera_width:int
    camera_height:int
    num_images:int
    cascades:int

@dataclass
class OccupancyDensityGrid:
    # float32, full-precision density values
    density: jax.Array
    # bool, a non-compact representation of the occupancy bitfield
    occ_mask: jax.Array
    # uint8, each bit is an occupancy value of a grid cell
    occupancy: jax.Array

    # uint32, indices of the grids that are alive (trainable)
    alive_indices: jax.Array

    # list of `int`s, upper bound of each cascade
    alive_indices_offset: List[int]=struct.field(pytree_node=False)

    @classmethod
    def create(cls, cascades: int, grid_resolution: int=128):
        """
        Inputs:
            cascades: number of cascades, paper: 𝐾 = 1 for all synthetic NeRF scenes (single grid)
                      and 𝐾 ∈ [1, 5] for larger real-world scenes (up to 5 grids, depending on scene
                      size)
            grid_resolution: resolution of the occupancy grid, the NGP paper uses 128.

        Example usage:
            ogrid = OccupancyDensityGrid.create(cascades=5, grid_resolution=128)
        """
        G3 = grid_resolution**3
        n_grids = cascades * G3
        occupancy = 255 * jnp.ones(
            shape=(n_grids // 8,),  # each bit is an occupancy value
            dtype=jnp.uint8,
        )
        density = jnp.zeros(
            shape=(n_grids,),
            dtype=jnp.float32,
        )
        occ_mask = jnp.zeros(
            shape=(n_grids,),
            dtype=jnp.bool_,
        )
        return cls(
            density=density,
            occ_mask=occ_mask,
            occupancy=occupancy,
            alive_indices=jnp.arange(n_grids, dtype=jnp.uint32),
            alive_indices_offset=np.cumsum([0] + [G3] * cascades).tolist(),
        )

    def mean_density_up_to_cascade(self, cas: int) -> Union[float, jax.Array]:
        return self.density[self.alive_indices[:self.alive_indices_offset[cas]]].mean()

class NeRFState(TrainState):
    # WARN:
    #   do not annotate fields with jax.Array as members with flax.truct.field(pytree_node=False),
    #   otherwise weird issues happen, e.g. jax tracer leak, array-to-boolean conversion exception
    #   while calling a jitted function with no helpful traceback.
    ogrid:OccupancyDensityGrid
    options:StateOptions=struct.field(pytree_node=False)
    images:jax.Array
    transform_matrices:jax.Array
    bg:bool
    nerf_fn:Callable=struct.field(pytree_node=False)
    bg_fn:Callable=struct.field(pytree_node=False)

    @classmethod
    def create(cls, *args, **kwargs):
        return super().create(apply_fn=None, *args, **kwargs)

    def __post_init__(self):
        assert self.apply_fn is None

    def update_ogrid_density(
        self,
        KEY: jran.KeyArray,
        cas: int,
        update_all: bool,
        max_inference: int,
    ) -> "NeRFState":
        G3 = self.options.density_grid_res**3
        cas_slice = slice(cas * G3, (cas + 1) * G3)
        cas_alive_indices = self.ogrid.alive_indices[self.ogrid.alive_indices_offset[cas]:self.ogrid.alive_indices_offset[cas+1]]
        aligned_indices = cas_alive_indices % G3  # values are in range [0, G3)
        n_grids = aligned_indices.shape[0]

        decay = .95
        cas_occ_mask = self.ogrid.occ_mask[cas_slice]
        cas_density_grid = self.ogrid.density[cas_slice].at[aligned_indices].set(self.ogrid.density[cas_slice][aligned_indices] * decay)

        if update_all:
            # During the first 256 training steps, we sample M = K * 128^{3} cells uniformly without
            # repetition.
            cas_updated_indices = aligned_indices
        else:
            M = max(1, n_grids // 2)
            # The first M/2 cells are sampled uniformly among all cells.
            KEY, key_firsthalf, key_secondhalf = jran.split(KEY, 3)
            indices_firsthalf = jran.choice(
                key=key_firsthalf,
                a=aligned_indices,
                shape=(max(1, M//2),),
                replace=True,  # allow duplicated choices
            )
            # Rejection sampling is used for the remaining samples to restrict selection to cells
            # that are currently occupied.
            # NOTE: Below is just uniformly sampling the occupied cells, not rejection sampling.
            cas_alive_occ_mask = cas_occ_mask[aligned_indices]
            indices_secondhalf = jran.choice(
                key=key_secondhalf,
                a=aligned_indices,
                shape=(max(1, M//2),),
                replace=True,  # allow duplicated choices
                p=cas_alive_occ_mask.astype(jnp.float32),  # only care about occupied grids
            )
            cas_updated_indices = jnp.concatenate([indices_firsthalf, indices_secondhalf])

        coordinates = morton3d_invert(cas_updated_indices).astype(jnp.float32)
        coordinates = coordinates / (self.options.density_grid_res - 1) * 2 - 1  # in [-1, 1]
        mip_bound = min(self.options.scene_bound, 2**cas)
        half_cell_width = mip_bound / self.options.density_grid_res
        coordinates *= mip_bound - half_cell_width  # in [-mip_bound+half_cell_width, mip_bound-half_cell_width]
        # random point inside grid cells
        KEY, key = jran.split(KEY, 2)
        coordinates += jran.uniform(
            key,
            coordinates.shape,
            coordinates.dtype,
            minval=-half_cell_width,
            maxval=half_cell_width,
        )

        def compute(pos):
            return self.nerf_fn(
                {"params": self.locked_params},
                (pos, pos),
            )[0]
        compute_v = jax.jit(jax.vmap(compute, in_axes=0))

        new_densities = map(
            lambda coords_part: compute_v(coords_part).ravel(),
            jnp.array_split(coordinates, max(1, n_grids // (max_inference))),
        )
        new_densities = jnp.concatenate(list(new_densities))
        '''
        new_densities = map(
            lambda coords_part: jax.jit(self.nerf_fn)(
                {"params": self.locked_params},
                coords_part,
                None,
            )[0].ravel(),
            jnp.array_split(jax.lax.stop_gradient(coordinates), max(1, n_grids // (max_inference))),
        )
        new_densities = jnp.concatenate(list(new_densities))
        '''
        #print('cas shape', cas_density_grid[cas_updated_indices].shape)

        cas_density_grid = cas_density_grid.at[cas_updated_indices].set(
            jax.lax.stop_gradient(jnp.maximum(cas_density_grid[cas_updated_indices], new_densities))
        )
        new_ogrid = self.ogrid.replace(
            density=self.ogrid.density.at[cas_slice].set(cas_density_grid),
        )
        return self.replace(ogrid=new_ogrid)

    @jax.jit
    def threshold_ogrid(self) -> "NeRFState":
        mean_density = self.ogrid.mean_density_up_to_cascade(1)
        density_threshold = jnp.minimum(self.density_threshold_from_min_step_size, mean_density)
        occupied_mask, occupancy_bitfield = packbits(
            density_threshold=density_threshold,
            density_grid=self.ogrid.density,
        )
        new_ogrid = self.ogrid.replace(
            occ_mask=occupied_mask,
            occupancy=occupancy_bitfield,
        )
        return self.replace(ogrid=new_ogrid)

    '''
    def mark_untrained_density_grid(self) -> "NeRFState":
        G = self.raymarch.density_grid_res
        G3 = G*G*G
        n_grids = self.cascades * G3
        all_indices = jnp.arange(n_grids, dtype=jnp.uint32)
        level, pos_idcs = all_indices // G3, all_indices % G3
        mip_bound = jnp.minimum(2 ** level, self.scene_bound).astype(jnp.float32)
        cell_width = 2 * mip_bound / G
        grid_xyzs = morton3d_invert(pos_idcs).astype(jnp.float32)  # [G3, 3]
        grid_xyzs /= G  # in range [0, 1)
        grid_xyzs -= 0.5  # in range [-0.5, 0.5)
        grid_xyzs *= 2 * mip_bound[:, None]  # in range [-mip_bound, mip_bound)
        vertex_offsets = cell_width[:, None, None] * jnp.asarray([
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
        ], dtype=jnp.float32)
        all_grid_vertices = grid_xyzs[:, None, :] + vertex_offsets

        @jax.jit
        def mark_untrained_density_grid_single_frame(
            alive_marker: jax.Array,
            transform_cw: jax.Array,
            grid_vertices: jax.Array,
        ):
            rot_cw, t_cw = transform_cw[:3, :3], transform_cw[:3, 3]
            # p_world, p_cam, T: [3, 1]
            # rot_cw: [3, 3]
            # p_world = rot_cw @ p_cam + t_cw
            p_aligned = grid_vertices - t_cw
            p_cam = (p_aligned[..., None, :] * rot_cw.T).sum(-1)

            # camera looks along the -z axis
            in_front_of_camera = p_cam[..., -1] < 0

            u, v = jnp.split(p_cam[..., :2] / (-p_cam[..., -1:]), [1], axis=-1)

            same_ray = True

            uv = jnp.concatenate([
                u * self.camera_fx + self.camera_cx,
                v * self.camera_fy + self.camera_cy,
            ], axis=-1)
            uv = uv / jnp.asarray([self.camera_width, self.camera_height], dtype=jnp.float32)

            within_frame_range = (uv >= 0.) & (uv < 1.)
            within_frame_range = (
                within_frame_range  # shape is [n_grids, 8, 2]
                    .all(axis=-1)  # u and v must both be within frame
            )

            visible_by_camera = (in_front_of_camera & within_frame_range & same_ray).any(axis=-1)  # grid should be trained if any of its 8 vertices is visible

            return alive_marker | visible_by_camera

        # cam_t = np.asarray(list(map(
        #     lambda frame: frame.transform_matrix_numpy[:3, 3],
        #     self.scene_meta.frames,
        # )))
        # np.savetxt("cams.xyz", cam_t)

        alive_marker = jnp.zeros(n_grids, dtype=jnp.bool_)
        for frame in self.scene_meta.frames):
            new_alive_marker_parts = map(
                lambda alive_marker_part, grid_vertices_part: mark_untrained_density_grid_single_frame(
                    alive_marker=alive_marker_part,
                    transform_cw=frame.transform_matrix_jax_array,
                    grid_vertices=grid_vertices_part,
                ),
                jnp.array_split(alive_marker, self.cascades),  # alive_marker_part
                jnp.array_split(all_grid_vertices, self.cascades),  # grid_vertices_part
            )
            alive_marker = jnp.concatenate(list(new_alive_marker_parts), axis=0)
            n_alive_grids = alive_marker.sum()
            ratio_trainable = n_alive_grids / n_grids
            if n_alive_grids == n_grids:
                break

        marked_density = jnp.where(alive_marker, self.ogrid.density, -1.)
        marked_occ_mask, marked_occupancy = packbits(
            density_threshold=min(self.density_threshold_from_min_step_size, self.ogrid.mean_density_up_to_cascade(1)) if self.step > 0 else -.5,
            density_grid=marked_density
        )

        # rgb = jnp.stack([~marked_occ_mask, jnp.zeros_like(marked_occ_mask, dtype=jnp.float32), marked_occ_mask]).T
        # xyzrgb = np.asarray(jnp.concatenate([grid_xyzs, rgb], axis=-1))
        # np.savetxt("blue_for_trainable.txt", xyzrgb)
        # np.savetxt("trainable.txt", xyzrgb[np.where(marked_occ_mask)])
        # np.savetxt("untrainable.txt", xyzrgb[np.where(~marked_occ_mask)])

        return self.replace(
            ogrid=self.ogrid.replace(
                density=marked_density,
                occ_mask=marked_occ_mask,
                occupancy=marked_occupancy,
                alive_indices=all_indices[alive_marker],
                alive_indices_offset=np.cumsum([0] + list(map(
                    lambda cas_alive_marker: int(cas_alive_marker.sum()),
                    jnp.split(alive_marker, self.cascades),
                ))).tolist(),
            ),
        )
    '''
        
    def epoch(self, iters: int) -> int:
        return self.step // iters

    @property
    def density_threshold_from_min_step_size(self) -> float:
        return .01 * self.options.diagonal_n_steps / (2 * min(self.options.scene_bound, 1) * 3**.5)

    @property
    def use_background_model(self) -> bool:
        return self.bg and self.params.get("bg") is not None

    @property
    def locked_params(self):
        return jax.lax.stop_gradient(self.params)

    @property
    def update_ogrid_interval(self) -> int:
        return min(16, self.step // 16 + 1)

    @property
    def should_call_update_ogrid(self) -> bool:
        return (
            self.step > 0
            and self.step % self.update_ogrid_interval == 0
        )

    @property
    def should_update_all_ogrid_cells(self) -> bool:
        return self.step < 256

    @property
    def should_write_batch_metrics(self) -> bool:
        return self.step % 16 == 0