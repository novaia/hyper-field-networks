# XLA Experimental

The purpose of this directory is to experiment with calling JAX generated XLA HLO from C++ in order to cut out Python overhead and speedup loops.

## XLA Extension

Download pre-compiled XLA extension:
```
wget https://github.com/elixir-nx/xla/releases/download/v0.6.0/xla_extension-x86_64-linux-gnu-cuda120.tar.gz
```

Extract it:
```
tar -xvzf ./xla_extension-x86_64-linux-gnu-cuda120.tar.gz
```

Delete the archive file:
```
rm ./xla_extension-x86_64-linux-gnu-cuda120.tar.gz
```

Compile execute_hlo.cpp

```
cmake . -B build
```
```
make -C build
```

## PJRT Runtime

The XLA extension process defined above didn't work, so I'm going to try building the PJRT runtime from the main XLA repository.

Install Bazelisk:
```
wget https://github.com/bazelbuild/bazelisk/releases/download/v1.18.0/bazelisk-linux-amd64 && \
chmod +x bazelisk-linux-amd64 && \
mv bazelisk-linux-amd64 /usr/local/bin/bazel
```

Symlink python3 to python so bazel build can find it:
```
ln -s /usr/bin/python3 /usr/bin/python
```

Install git, gcc-10, and g++-10:
```
apt-get update && apt-get install -y git gcc-10 g++-10 && apt-get clean \ && rm -rf /var/lib/apt/lists/*
```

Clone XLA repository:
```
git clone https://github.com/openxla/xla.git
```

Change into XLA directory:
```
cd xla
```

Configure build:
```
yes '' | GCC_HOST_COMPILER_PATH=/usr/bin/gcc-10 CC=/usr/bin/gcc-10 TF_NEED_ROCM=0 TF_NEED_CUDA=1 TF_CUDA_CLANG=0 ./configure
```

Build PJRT runtime:
```
bazel build --strip=never //xla/pjrt/c:pjrt_c_api_cpu
```