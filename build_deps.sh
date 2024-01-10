# Build and install tiny-cuda-nn.
rm -f /usr/lib/libtiny-cuda-nn.a &&
cd dependencies/tiny-cuda-nn &&
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo &&
cmake --build build --config RelWithDebInfo -j &&
ln -s /project/dependencies/tiny-cuda-nn/build/libtiny-cuda-nn.a /usr/lib/libtiny-cuda-nn.a &&

# Install serde-helper.
rm -f /usr/local/include/serde-helper/serde.h &&
cd .. &&
mkdir -p /usr/local/include/serde-helper/ &&
ln -s /project/dependencies/serde-helper/serde.h /usr/local/include/serde-helper/serde.h

# Install jax extensions.
python3 -m pip install ./volume-rendering-jax ./jax-tcnn
