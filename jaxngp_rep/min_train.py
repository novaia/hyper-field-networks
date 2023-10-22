# Stripped down version of jaxngp train.py
import dataclasses
import functools
import gc
import time
from typing import Any, Dict, List, Tuple, Union
from tqdm import tqdm

import jax
import jax.numpy as jnp
import jax.random as jran
import optax

from volrendjax import morton3d_invert
from nerfs import make_nerf_ngp, make_skysphere_background_model_ngp
from utils import common, data
from utils.types import (
    NeRFState,
    OccupancyDensityGrid,
    RenderedImage,
    RigidTransformation,
    SceneData,
    SceneOptions,
    RayMarchingOptions,
    RenderingOptions
)
from cuda import render_rays_train

def train_epoch(
    KEY: jran.KeyArray,
    state: NeRFState,
    scene: SceneData,
    iters: int,
    total_samples: int,
    ep_log: int,
    total_epochs: int
) -> Tuple[NeRFState, Dict[str, Any]]:
    n_processed_rays = 0
    total_loss = None
    interrupted = False

    try:
        with tqdm(range(iters), desc="Training epoch#{:03d}/{:d}".format(ep_log, total_epochs)) as pbar:
            start = int(state.step) % iters
            pbar.update(start)
            for _ in range(start, iters):
                KEY, key_perm, key_train_step = jran.split(KEY, 3)
                perm = jran.choice(key_perm, scene.n_pixels, shape=(total_samples,), replace=True)
                state, metrics = train_step(
                    state,
                    KEY=key_train_step,
                    total_samples=total_samples,
                    scene=scene,
                    perm=perm,
                )
                n_processed_rays += metrics["n_valid_rays"]
                loss = metrics["loss"]
                if total_loss is None:
                    total_loss = loss
                else:
                    total_loss = jax.tree_util.tree_map(
                        lambda total, new: total + new * metrics["n_valid_rays"],
                        total_loss,
                        loss,
                    )

                pbar.set_description_str(
                    desc="Training epoch#{:03d}/{:d} ".format(
                        ep_log,
                        total_epochs,
                    ) + format_metrics(metrics),
                )
                pbar.update(1)

                if state.should_call_update_ogrid:
                    # update occupancy grid
                    for cas in range(state.scene_meta.cascades):
                        KEY, key = jran.split(KEY, 2)
                        state = state.update_ogrid_density(
                            KEY=key,
                            cas=cas,
                            update_all=bool(state.should_update_all_ogrid_cells),
                            max_inference=total_samples,
                        )
                    state = state.threshold_ogrid()
    except (InterruptedError, KeyboardInterrupt):
        interrupted = True

    return state, {
        "total_loss": total_loss,
        "n_processed_rays": n_processed_rays,
        "interrupted": interrupted,
    }


def train():
    KEY = jran.PRNGKey(0)
    scene_options = SceneOptions(
        sharpness_threshold=-1.,
        resolution_scale=1.0,
        camera_near=0.3,
        max_mem_mbytes=2500,
        bound=1
    )
    raymarching_options = RayMarchingOptions(
        diagonal_n_steps=1024,
        perturb=True,
        density_grid_res=128
    )
    rendering_options = RenderingOptions(
        bg=(1., 1., 1.), # Ignored.
        random_bg=True
    )
    # args.train.tv_scale
    tv_scale = 0.01
    lr = 1e-2
    train_iters = 1000
    train_epochs = 1
    bs = 256 * 1024

    frames_train = ['data/lego']
    scene_train = data.load_scene(srcs=frames_train, scene_options=scene_options)
    scene_meta = scene_train.meta
    print('Bound:', scene_meta.bound)
    print('Bg:', scene_meta.bg)
    print('Camera:', scene_meta.camera)
    print('Cascades:', scene_meta.cascades)

    nerf_model, init_input = (
        make_nerf_ngp(bound=scene_meta.bound, inference=False, tv_scale=tv_scale),
        (
            jnp.zeros((1, 3), dtype=jnp.float32),
            jnp.zeros((1, 3), dtype=jnp.float32),
            jnp.zeros((1, scene_meta.n_extra_learnable_dims), dtype=jnp.float32),
        )
    )
    KEY, key = jran.split(KEY, 2)
    nerf_variables = nerf_model.init(key, *init_input)
    #print(nerf_model.tabulate(key, *init_input))
    
    KEY, key = jran.split(KEY, 2)
    # training state
    state = NeRFState.create(
        ogrid=OccupancyDensityGrid.create(
            cascades=scene_meta.cascades,
            grid_resolution=raymarching_options.density_grid_res,
        ),
        raymarch=raymarching_options,
        render=rendering_options,
        scene_options=scene_options,
        scene_meta=scene_meta,
        # unfreeze the frozen dict so that the weight_decay mask can apply, see:
        #   <https://github.com/deepmind/optax/issues/160>
        #   <https://github.com/google/flax/issues/1223>
        nerf_fn=nerf_model.apply,
        #bg_fn=bg_model.apply if scene_meta.bg else None,
        bg_fn=None,
        params={
            # It looks like flax doesn't freeze params anymore.
            #"nerf": nerf_variables["params"].unfreeze(),
            "nerf": nerf_variables["params"],
            #"bg": bg_variables["params"].unfreeze() if scene_meta.bg else None,
            "bg": None,
            "appearance_embeddings": jran.uniform(
                key=key,
                shape=(len(scene_meta.frames), scene_meta.n_extra_learnable_dims),
                dtype=jnp.float32,
                minval=-1,
                maxval=1,
            ),
        },
        tx=make_optimizer(lr),
    )
    # TODO: see how not doing this affects the training.
    state = state.mark_untrained_density_grid()

    # training loop
    for ep in range(state.epoch(train_iters), train_epochs):
        gc.collect()
        ep_log = ep + 1

        KEY, key_resample, key_train = jran.split(KEY, 3)
        scene_train = scene_train.resample_pixels(
            KEY=key_resample,
            new_max_mem_mbytes=scene_options.max_mem_mbytes,
        )

        state, metrics = train_epoch(
            KEY=key_train,
            state=state,
            scene=scene_train,
            iters=train_iters,
            total_samples=bs,
            ep_log=ep_log,
            total_epochs=train_epochs,
        )
    jnp.save('data/occupancy_grid_density.npy', state.ogrid.occ_mask.astype(jnp.float32))
    occupancy_grid_coordinates = morton3d_invert(
        jnp.arange(state.ogrid.occ_mask.shape[0], dtype=jnp.uint32)
    )
    occupancy_grid_coordinates = occupancy_grid_coordinates / (128 - 1) * 2 - 1
    jnp.save('data/occupancy_grid_coordinates.npy', occupancy_grid_coordinates)

def make_optimizer(lr: float) -> optax.GradientTransformation:
    lr_sch = optax.exponential_decay(
        init_value=lr,
        transition_steps=10_000,
        decay_rate=1/3,  # decay to `1/3 * init_lr` after `transition_steps` steps
        staircase=True,  # use integer division to determine lr drop step
        transition_begin=10_000,  # hold the initial lr value for the initial 10k steps (but first lr drop happens at 20k steps because `staircase` is specified)
        end_value=lr / 100,  # stop decaying at `1/100 * init_lr`
    )
    optimizer_network = optax.adam(
        learning_rate=lr_sch,
        b1=0.9,
        b2=0.99,
        # paper:
        #   the small value of 𝜖 = 10^{−15} can significantly accelerate the convergence of the
        #   hash table entries when their gradients are sparse and weak.
        eps=1e-15,
        eps_root=1e-15,
    )
    optimizer_ae = optax.adam(
        learning_rate=1e-4,
        b1=.9,
        b2=.99,
        eps=1e-8,
        eps_root=0,
    )
    return optax.chain(
        optax.multi_transform(
            transforms={
                "network": optimizer_network,
                "ae": optimizer_ae,
            },
            param_labels={
                "nerf": "network",
                "bg": "network",
                "appearance_embeddings": "ae",
            },
        ),
        optax.add_decayed_weights(
            # In NeRF experiments, the network can converge to a reasonably low loss during the
            # first ~50k training steps (with 1024 rays per batch and 1024 samples per ray), but the
            # loss becomes NaN after about 50~150k training steps.
            # paper:
            #   To prevent divergence after long training periods, we apply a weak L2 regularization
            #   (factor 10^{−6}) to the neural network weights, ...
            weight_decay=1e-6,
            # paper:
            #   ... to the neural network weights, but not to the hash table entries.
            mask={
                "nerf": {
                    "density_mlp": True,
                    "rgb_mlp": True,
                    "position_encoder": False,
                },
                "bg": True,
                "appearance_embeddings": False,
            },
        ),
    )


@common.jit_jaxfn_with(
    static_argnames=["total_samples"],
    donate_argnums=(0,),  # NOTE: this only works for positional arguments, see <https://jax.readthedocs.io/en/latest/faq.html#buffer-donation>
)
def train_step(
    state: NeRFState,
    /,
    KEY: jran.KeyArray,
    total_samples: int,
    scene: SceneData,
    perm: jax.Array,
) -> Tuple[NeRFState, Dict[str, Union[jax.Array, float]]]:
    # indices of views and pixels
    view_idcs, pixel_idcs = scene.get_view_indices(perm), scene.get_pixel_indices(perm)

    # TODO:
    #   merge this and `models.renderers.make_rays_worldspace` as a single function
    def make_rays_worldspace() -> Tuple[jax.Array, jax.Array]:
        # [N], [N]
        x, y = (
            jnp.mod(pixel_idcs, scene.meta.camera.width),
            jnp.floor_divide(pixel_idcs, scene.meta.camera.width),
        )
        # [N, 3]
        d_cam = scene.meta.camera.make_ray_directions_from_pixel_coordinates(x, y, use_pixel_center=True)

        # [N, 3]
        o_world = scene.transforms[view_idcs, -3:]

        # [N, 3, 3]
        R_cws = scene.transforms[view_idcs, :9].reshape(-1, 3, 3)
        # [N, 3]
        # equavalent to performing `d_cam[i] @ R_cws[i].T` for each i in [0, N)
        d_world = (d_cam[:, None, :] * R_cws).sum(-1)

        return o_world, d_world

    # CAVEAT: gradient is only calculate w.r.t. the first parameter of this function
    # (`params_to_optimize here`), any parameters that need to be optimized should be taken from
    # this parameter, instead from the outer-scope `state.params`.
    def loss_fn(params_to_optimize, gt_rgba_f32, KEY):
        o_world, d_world = make_rays_worldspace()
        appearance_embeddings = (
            params_to_optimize["appearance_embeddings"][view_idcs]
                if "appearance_embeddings" in params_to_optimize
                else jnp.empty(0)
        )
        if state.use_background_model:
            bg = state.bg_fn(
                {"params": params_to_optimize["bg"]},
                o_world,
                d_world,
                appearance_embeddings,
            )
        elif state.render.random_bg:
            KEY, key = jran.split(KEY, 2)
            bg = jran.uniform(key, shape=(o_world.shape[0], 3), dtype=jnp.float32, minval=0, maxval=1)
        else:
            bg = jnp.asarray(state.render.bg)
        KEY, key = jran.split(KEY, 2)
        batch_metrics, pred_rgbds, tv = render_rays_train(
            KEY=key,
            o_world=o_world,
            d_world=d_world,
            appearance_embeddings=appearance_embeddings,
            bg=bg,
            total_samples=total_samples,
            state=state.replace(params=params_to_optimize),
        )
        pred_rgbs, pred_depths = jnp.array_split(pred_rgbds, [3], axis=-1)
        gt_rgbs = data.blend_rgba_image_array(imgarr=gt_rgba_f32, bg=bg)
        batch_metrics["loss"] = {
            "rgb": jnp.where(
                batch_metrics["ray_is_valid"],
                optax.huber_loss(pred_rgbs, gt_rgbs, delta=0.1).mean(axis=-1),
                0.,
            ).sum() / batch_metrics["n_valid_rays"],
            "total_variation": tv,
        }
        loss = jax.tree_util.tree_reduce(lambda x, y: x + y, batch_metrics["loss"])
        return loss, batch_metrics

    loss_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    KEY, key = jran.split(KEY, 2)
    (_, batch_metrics), grads = loss_grad_fn(
        state.params,
        scene.rgbas_u8[perm].astype(jnp.float32) / 255,
        key,
    )
    state = state.apply_gradients(grads=grads)
    return state, batch_metrics


def format_metrics(metrics: Dict[str, Union[jax.Array, float]]) -> str:
    loss = metrics["loss"]
    return "batch_size={}/{} samp./ray={:.1f}/{:.1f} n_rays={} loss:{{rgb={:.2e}({:.2f}dB),tv={:.2e}}}".format(
        metrics["measured_batch_size"],
        metrics["measured_batch_size_before_compaction"],
        metrics["measured_batch_size"] / metrics["n_valid_rays"],
        metrics["measured_batch_size_before_compaction"] / metrics["n_valid_rays"],
        metrics["n_valid_rays"],
        loss["rgb"],
        data.linear_to_db(loss["rgb"], maxval=1.),
        loss["total_variation"],
    )

if __name__ == "__main__":
    train()

