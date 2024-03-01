JAX_PACKAGE_URL="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
python3 -m pip install "jax[cuda11_cudnn86]" -f $JAX_PACKAGE_URL

DALI_PACKAGE_URL="https://developer.download.nvidia.com/compute/redist"
python3 -m pip install -y \
    --extra-index-url $DALI_PACKAGE_URL \
    --upgrade nvidia-dali-cuda120

python3 -m pip install requirements.txt
