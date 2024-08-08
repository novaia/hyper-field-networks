{
    description = "float stuff";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/24.05";
        flake-utils.url = "github:numtide/flake-utils";
        # Patched version of nixGL from kenrandunderscore.
        # PR: https://github.com/nix-community/nixGL/pull/165
        # TODO: switch back to github:nix-community/nixGL when PR is merged.
        nixgl.url = "github:hayden-donnelly/nixGL";
    };
    outputs = inputs@{ 
        self, 
        nixpkgs, 
        flake-utils, 
        nixgl, 
        ... 
    }: flake-utils.lib.eachSystem [ "x86_64-linux" ] (system: 
        let
            inherit (nixpkgs) lib;
            overlays = [ nixgl.overlay ];
            pkgs = import nixpkgs {
                inherit system overlays;
                config = {
                    allowUnfree = true;
                    cudaSupport = true;
                };
            };
        in {
            devShells.default = (with pkgs; mkShell.override { stdenv = gcc12Stdenv; }) {
            #pkgs.mkShell { #let gcc = pkgs.gcc12; in pkgs.mkShell {
                name = "float-tokenization";
                buildInputs = with pkgs; [
                    cudaPackages.cudatoolkit
                    gcc12
                ];
                shellHook = ''
                    export CUDA_PATH=${pkgs.cudatoolkit}
                    source <(sed -Ee '/\$@/d' ${lib.getExe pkgs.nixgl.nixGLIntel})
                    source <(sed -Ee '/\$@/d' ${lib.getExe pkgs.nixgl.auto.nixGLNvidia}*)
                '';
            };
        }
    );
}
