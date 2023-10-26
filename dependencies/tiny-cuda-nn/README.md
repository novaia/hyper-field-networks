# Tiny CUDA Neural Networks

This is a stripped down version of tiny-cuda-nn containing only the parts required 
for jax-tcnn. The official version can be found at [NVLabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).

To build the project, use the following commands:
```
tiny-cuda-nn$ cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```
```
tiny-cuda-nn$ cmake --build build --config RelWithDebInfo -j
```

## Citation
```bibtex
@software{tiny-cuda-nn,
	author = {M\"uller, Thomas},
	license = {BSD-3-Clause},
	month = {4},
	title = {{tiny-cuda-nn}},
	url = {https://github.com/NVlabs/tiny-cuda-nn},
	version = {1.7},
	year = {2021}
}
```