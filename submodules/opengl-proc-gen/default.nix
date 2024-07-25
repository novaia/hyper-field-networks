{ 
    lib,
    stdenv,
    version, 
    symlinkJoin, 
    buildPythonPackage,
    setuptools-scm,
    cmake,
    ninja,
    pybind11,
    python3,
    gcc12,
    glad,
    glfw,
    libpng
}:

buildPythonPackage rec {
    pname = "opengl-proc-gen";
    inherit version;
    src = ./.;
    format = "pyproject";
    dontUseCmakeConfigure = true;
    
    nativeBuildInputs = [
        cmake
        ninja
        pybind11
        setuptools-scm
        gcc12
        glfw
        glad
        libpng
    ];

    buildInputs = [
        stdenv.cc.cc.lib # Required for libstdc++.so
    ];

    preConfigure = ''
        export CC=${gcc12}/bin/gcc
        export CXX=${gcc12}/bin/g++
    '';

    preFixup = ''
        patchelf --set-rpath "${lib.makeLibraryPath buildInputs}" \
            $out/lib/python${python3.pythonVersion}/site-packages/opengl_proc_gen/*.so
    '';

    doCheck = false;
    pythonImportsCheck = [ "opengl_proc_gen" ];
}
