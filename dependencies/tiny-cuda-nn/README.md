# Tiny CUDA Neural Networks

This is a stripped down version of tiny-cuda-nn containing only the parts required 
for jax-tcnn. The official version can be found at [NVLabs/tiny-cuda-nn](https://github.com/NVlabs/tiny-cuda-nn).

To build the library, use the following commands:
```
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
```
```
cmake --build build --config RelWithDebInfo -j
```
You can then link the static library to /usr/lib/ or wherever you want to install it.
```
ln -s /project/dependencies/tiny-cuda-nn/build/libtiny-cuda-nn.a /usr/lib/libtiny-cuda-nn.a
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