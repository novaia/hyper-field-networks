{ pkgs }:
pkgs.stdenv.mkDerivation {
    pname = "glad";
    version = "v2.0.4";
    src = ./.;
    buildPhase = ''
        mkdir -p $out/include
        cp -r ./include/* $out/include
        gcc -std=c99 -c ./src/gl.c -I $out/include
        ar rcs libglad.a gl.o
    '';
    installPhase = ''
        touch libglad.a
        mkdir -p $out/lib
        cp libglad.a $out/lib
    '';
}
