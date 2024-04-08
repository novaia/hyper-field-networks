{ 
    stdenv,
    lib, 
    version, 
    symlinkJoin, 
    buildPythonPackage,
    setuptools-scm,
    cmake,
    pybind11,
    fmt,
    python3,
    serde-helper,
    cudatoolkit,
    jax,
    jaxlib,
    gcc12
}:

buildPythonPackage rec {
    pname = "float-tokenization";
    inherit version;
    src = ./.;

    format = "pyproject";
    
    CUDA_HOME = cudatoolkit;

    nativeBuildInputs = [
        cmake
        pybind11
        setuptools-scm
        gcc12
    ];
    dontUseCmakeConfigure = true;

    buildInputs = [
        serde-helper
        cudatoolkit
        fmt
        #stdenv.cc.cc.lib
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
            $out/lib/python${python3.pythonVersion}/site-packages/float_tokenization/*.so
    '';

    doCheck = false;
    pythonImportsCheck = [ "float_tokenization" ];
}
