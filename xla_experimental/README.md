# XLA Experimental

The purpose of this directory is to experiment with calling JAX generated XLA HLO from C++ in order to cut out Python overhead and speedup loops.

## XLA Extension Setup

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

## Compiling execute_hlo.cpp

```
cmake . -B build
```

```
make -C build
```