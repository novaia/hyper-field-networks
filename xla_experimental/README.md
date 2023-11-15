# XLA Experimental

The purpose of this directory is to experiment with calling JAX generated XLA HLO from C++ in order to cut out Python overhead and speedup loops.

## Executing HLO with PJRT Runtime

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

Build execute_hlo with XLA dependencies:
```
bazel build //:execute_hlo
```

Run execute_hlo:
```
bazel-bin/execute_hlo
```
