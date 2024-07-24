{
    description = "Synthetic 3D data creation with OpenGL";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/23.11";
        flake-utils.url = "github:numtide/flake-utils";
    };
    outputs = inputs@{ self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system: let
        inherit (nixpkgs) lib;
        pkgs = import nixpkgs {
            inherit system;
            config = {
                allowUnfree = true;
                cudaSupport = true;
            };
        };
    in {
        devShells = let
            glad = import ../../external/glad { inherit pkgs; };
        in rec {
            default = pkgs.mkShell {
                buildInputs = with pkgs; [
                    libpng
                    glfw
                    glad
                    clang-tools
                ];
                dot_clangd = ''
                    CompileFlags:                     # Tweak the parse settings
                        Add:
                            - "-I${pkgs.libpng}/include
                            - "-I${glad}/include
                            - "-I${pkgs.glfw}/include
                    Remove: "-W*"                   # strip all other warning-related flags
                    Compiler: "clang++"             # Change argv[0] of compile flags to clang++
                    # vim: ft=yaml:
                '';
                shellHook = ''
                    echo "use \`echo \$dot_clangd >.clangd\` for development"
                    [[ "$-" == *i* ]] && exec "$SHELL"
                '';
            };
        };
        packages = {
            default = pkgs.callPackage ./. {};
        };
    });
}
