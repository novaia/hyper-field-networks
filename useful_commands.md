Synthesizing NeRF datasets with a specified Blender script:
```
python3 -m synthdata --script alien.py --number 20 --save_directory data/renders/alien --num_renders 200
```

Generating NGP NeRFs from a directory of datasets:
```
python3 scripts/generate_ngp_nerfs.py --dataset_dir data/synthetic_nerf_data --output_dir data/synthetic_nerfs
```

Training Linear Transformer DDIM with high memory usage:
```
XLA_PYTHON_CLIENT_MEM_FRACTION=.97 python3 -m hypernets.sliding_ltd
```

Generating dataset of CIFAR10 NGP Images:
```
python3 -m fields.train_multiple.ngp_image --config_path configs/ngp_image.json --input_path data/CIFAR10 --output_path data/ngp_images/ngp_cifar10
```

Converting PyTorch Diffusers model to Flax:
```
python3 -m diffusion.hf_diffusers.torch_to_flax --config_path configs/stable_diffusion_2_unet.json --model_path data/models/stable_diffusion_2_unet.safetensors --output_path data/models/stable_diffusion_2_unet_flax.npy --model_type unet
```
