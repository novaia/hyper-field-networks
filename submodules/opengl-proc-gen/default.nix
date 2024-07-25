{ pkgs, glad }:
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
        gcc -std=c99 -c ./csrc/main.c ./csrc/file_io.c ./csrc/vector_matrix_math.c ./csrc/rendering.c
        gcc main.o file_io.o vector_matrix_math.o rendering.o -lglfw -lglad -lm -lpng -o 3d 
    '';
    installPhase = ''
        mkdir -p $out/bin
        cp 3d $out/bin
    '';
}
