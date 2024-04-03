{ pkgs }:
let 
    glad = import ./dependencies/glad { inherit pkgs; };
in
pkgs.stdenv.mkDerivation {
    pname = "3d";
    version = "v0.0.0";
    src = ./.;
    buildInputs = with pkgs; [ 
        glfw
        glad
        libpng
    ];
    buildPhase = ''
        bash shaders_to_header.sh
        gcc -std=c99 -c ./src/main.c ./src/file_io.c ./src/matrices.c ./src/rendering.c
        gcc main.o file_io.o matrices.o rendering.o -lglfw -lglad -lm -lpng -o 3d 
    '';
    installPhase = ''
        mkdir -p $out/bin
        cp 3d $out/bin
    '';
}
