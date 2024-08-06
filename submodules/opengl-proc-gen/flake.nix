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
        glad = import ../../external/glad { inherit pkgs; };
        single-header-libs = import ../../external/single-header-libs { inherit pkgs; };
    in {
        devShells = rec {
            default = pkgs.mkShell {
                buildInputs = with pkgs; [
                    libpng
                    glfw
                    glad
                    single-header-libs
                    clang-tools
                ];
            };
        };
        packages = {
            default = pkgs.callPackage ./. { inherit pkgs glad; };
        };
    });
}
