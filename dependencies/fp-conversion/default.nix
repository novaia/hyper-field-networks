{ 
    stdenv,
    lib, 
    version, 
    symlinkJoin, 
    buildPythonPackage,
    setuptools-scm,
    cmake,
    pybind11,
    python3,
    gcc12
}:

buildPythonPackage rec {
    pname = "fp-conversion";
    inherit version;
    src = ./.;

    format = "pyproject";

    nativeBuildInputs = [
        cmake
        pybind11
        setuptools-scm
    ];
    dontUseCmakeConfigure = true;

    buildInputs = [
        stdenv.cc.cc.lib
    ];

    preFixup = ''
        patchelf --set-rpath "${lib.makeLibraryPath buildInputs}" \
            $out/lib/python${python3.pythonVersion}/site-packages/fp_conversion/*.so
    '';

    doCheck = false;
    pythonImportsCheck = [ "fp_conversion" ];
}
