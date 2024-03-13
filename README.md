# g3dm
Experiments in generative 3D modelling.

## Running Tests
Use PyTest to run the tests in the tests directory:
```
pytest tests
```

## Development Environment
It is recommended to use the Docker development environment.
The Nix development environment is currently experimental.

### Nix
Entering development shell:
```
bash devshell.sh
```

### Docker
Building image:
```
bash docker_build.sh
```

Running the container:
```
bash docker_run.sh
```

To open VS Code inside the container, first make sure you have the Dev Containers extension installed and have the container running.
Next, open the root of the repository in VS Code and select ``Dev Containers: Attach to Running Container`` from the command palette,
then select the container you want to attach to. See [attach to a running container](https://code.visualstudio.com/docs/remote/attach-container) 
for more details.

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
