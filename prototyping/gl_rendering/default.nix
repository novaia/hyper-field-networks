{ pkgs }:
let 
    glad = import ./dependencies/glad { inherit pkgs; };
in
pkgs.stdenv.mkDerivation {
    pname = "synthetic3d";
    version = "v0.0.0";
    src = ./.;
    buildInputs = with pkgs; [ 
        glfw
        glad
        libpng
    ];
    buildPhase = ''
        bash shaders_to_header.sh
        gcc -std=c99 -c ./src/main.c ./src/file_io.c ./src/transform.c
        gcc main.o file_io.o transform.o -lglfw -lglad -lm -lpng -o synthetic3d
    '';
    installPhase = ''
        mkdir -p $out/bin
        cp synthetic3d $out/bin
    '';
}
