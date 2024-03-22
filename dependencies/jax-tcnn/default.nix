{ 
    lib, 
    version, 
    symlinkJoin, 
    linkFarm, 
    cudaCapabilities, 
    buildPythonPackage,
    setuptools-scm,
    cmake,
    ninja,
    pybind11,
    fmt,
    serde-helper,
    cudatoolkit,
    tiny-cuda-nn,
    python3,
    chex,
    jax,
    jaxlib
}:

let
    dropDot = x: builtins.replaceStrings ["."] [""] x;
    minGpuArch = let
        min = lhs: rhs: if (builtins.compareVersions lhs rhs) < 0
        then lhs
        else rhs;
    in dropDot (builtins.foldl' min "998244353" cudaCapabilities);
in buildPythonPackage rec {
    pname = "jax-tcnn";
    inherit version;
    src = ./.;

    format = "pyproject";

    CUDA_HOME = cudatoolkit;

    nativeBuildInputs = [
        cmake
        ninja
        pybind11
        setuptools-scm
    ];
    dontUseCmakeConfigure = true;
    cmakeFlags = [
        "-DTCNN_MIN_GPU_ARCH=${minGpuArch}"
        "-DCMAKE_CUDA_ARCHITECTURES=${lib.concatStringsSep ";" (map dropDot cudaCapabilities)}"
    ];

    buildInputs = [
        cudatoolkit
        fmt
        serde-helper
        tiny-cuda-nn
    ];

    propagatedBuildInputs = [
        chex
        jax
        jaxlib
    ];

    #preFixup = ''
    #    patchelf --set-rpath "${lib.makeLibraryPath buildInputs}/lib" $out/lib/python${python3.pythonVersion}/site-packages/jaxtcnn/*.so
    #'';

    doCheck = false;
    pythonImportsCheck = [ "jaxtcnn" ];
}
