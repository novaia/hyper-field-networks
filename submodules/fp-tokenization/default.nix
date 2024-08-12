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
    cudatoolkit,
    python3,
    jax,
    jaxlib,
    gcc12
}:

buildPythonPackage rec {
    pname = "fp-tokenization";
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
        cudatoolkit
        fmt
    ];

    propagatedBuildInputs = [
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
            $out/lib/python${python3.pythonVersion}/site-packages/fp_tokenization/*.so
    '';

    doCheck = false;
    pythonImportsCheck = [ "fp_tokenization" ];
}
