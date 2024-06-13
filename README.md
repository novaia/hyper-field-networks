# g3dm
Experiments in generative 3D modelling.

## Development Environment
This project's development environment is managed with Nix. You can follow the steps below to get started.
1. Install Nix with the [official installer](https://nixos.org/download/) or the [determinate installer](https://github.com/DeterminateSystems/nix-installer).
2. Enable the experimental Nix Flakes feature by adding the following line to ``~/.config/nix/nix.conf`` or ``/etc/nix/nix.conf`` 
(this step can be skipped if you installed nix with the [determinate installer](https://github.com/DeterminateSystems/nix-installer)).
```
experimental-features = nix-command flakes
```
3. Run the following command to open a development shell with all the dependencies installed.
```
nix develop --impure
```

## Citations
```bibtex
@article{mueller2022instant,
    author={Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title={Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal={ACM Trans. Graph.},
    issue_date={July 2022},
    volume={41},
    number={4},
    month=jul,
    year={2022},
    pages={102:1--102:15},
    articleno={102},
    numpages={15},
    url={https://doi.org/10.1145/3528223.3530127},
    doi={10.1145/3528223.3530127},
    publisher={ACM},
    address={New York, NY, USA},
}
```

```bibtex
@software{tiny-cuda-nn,
    author={M\"uller, Thomas},
    license={BSD-3-Clause},
    month={4},
    title={{tiny-cuda-nn}},
    url={https://github.com/NVlabs/tiny-cuda-nn},
    version={1.7},
    year={2021}
}
```

```bibtex
@inproceedings{mildenhall2020nerf,
    title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
    author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
    year={2020},
    booktitle={ECCV},
}
```

```bibtex
@software{Zhang_jaxngp_2023,
    author={Zhang, Gaoyang and Chen, Yingxi},
    month=may,
    title={{jaxngp}},
    url={https://github.com/blurgyy/jaxngp},
    year={2023}
}
```

```bibtex
@misc{erkoç2023hyperdiffusion,
    title={HyperDiffusion: Generating Implicit Neural Fields with Weight-Space Diffusion}, 
    author={Ziya Erkoç and Fangchang Ma and Qi Shan and Matthias Nießner and Angela Dai},
    year={2023},
    eprint={2303.17015},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
```
