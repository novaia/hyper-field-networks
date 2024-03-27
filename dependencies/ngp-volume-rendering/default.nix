{ 
    lib, 
    version, 
    symlinkJoin, 
    buildPythonPackage,
    setuptools-scm,
    cmake,
    ninja,
    pybind11,
    fmt,
    serde-helper,
    cudatoolkit,
    python3,
    chex,
    jax,
    jaxlib,
    gcc12
}:

buildPythonPackage rec {
    pname = "ngp-volume-rendering";
    inherit version;
    src = ./.;

    format = "pyproject";

    CUDA_HOME = cudatoolkit;

    nativeBuildInputs = [
        cmake
        ninja
        pybind11
        setuptools-scm
        gcc12
    ];
    dontUseCmakeConfigure = true;

    buildInputs = [
        serde-helper
        cudatoolkit
        fmt
    ];

    propagatedBuildInputs = [
        chex
        jax
        jaxlib
    ];
    
    preConfigure = ''
        export CC=${gcc12}/bin/gcc
        export CXX=${gcc12}/bin/g++
        export CUDAHOSTCXX=${gcc12}/bin/g++
    '';

    preFixup = ''
        patchelf --set-rpath "${lib.makeLibraryPath buildInputs}" \
            $out/lib/python${python3.pythonVersion}/site-packages/ngp_volume_rendering/*.so
    '';

    doCheck = false;
    pythonImportsCheck = [ "ngp_volume_rendering" ];
}
