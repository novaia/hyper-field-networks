{ pkgs }:
pkgs.stdenv.mkDerivation {
    pname = "single-header-libs";
    version = "v0.1.0";
    src = ./.;
    buildPhase = ''
        mkdir -p $out/include
        cp -r ./include/* $out/include
    '';
}
