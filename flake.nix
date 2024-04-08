{
    description = "3D generative AI";

    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/23.11";
        nixpkgs-unstable.url = "github:nixos/nixpkgs/nixpkgs-unstable";
        flake-utils.url = "github:numtide/flake-utils";
        # Patched version of nixGL from kenrandunderscore.
        # PR: https://github.com/nix-community/nixGL/pull/165
        # TODO: switch back to github:nix-community/nixGL when PR is merged.
        nixgl.url = "github:hayden-donnelly/nixGL";
    };
    outputs = inputs@{ self, nixpkgs, nixpkgs-unstable, flake-utils, nixgl, ... }: 
    let 
        dependencies = import ./dependencies;
    in flake-utils.lib.eachSystem [ "x86_64-linux" ] (system: let
        inherit (nixpkgs-unstable) lib;
    in {
        devShells = let
            pyVer = "311";
            py = "python${pyVer}";
            overlays = [
                dependencies.overlay
                nixgl.overlay
                (final: prev: {
                    ${py} = prev.${py}.override {
                        packageOverrides = finalPkgs: prevPkgs: {
                            jax = prevPkgs.jax.overridePythonAttrs (o: {
                                nativeCheckInputs = [];
                                pythonImportsCheck = [];
                                pytestFlagsArray = [];
                                passthru.tests = [];
                                doCheck = false;
                            });
                            # For some reason Flax has jaxlib as a buildInput and tensorflow as a 
                            # nativeCheckInput, so set jaxlib to jaxlib-bin in order to avoid building 
                            # jaxlib and turn off all checks to avoid building tensorflow.
                            jaxlib = prevPkgs.jaxlib-bin;
                            flax = prevPkgs.flax.overridePythonAttrs (o: {
                                nativeCheckInputs = [];
                                pythonImportsCheck = [];
                                pytestFlagsArray = [];
                                doCheck = false;
                            });
                            wandb = prevPkgs.wandb.overridePythonAttrs(o: {
                                nativeCheckInputs = [];
                                pythonIMportsCheck = [];
                                doCheck = false;
                            });
                        };
                    };
                })
            ];
            unstableCudaPkgs = import nixpkgs-unstable {
                inherit system overlays;
                config = {
                    allowUnfree = true;
                    cudaSupport = true;
                };
            };
        in rec {
            default = unstableCudaPkgs.mkShell {
                name = "cuda";
                buildInputs = [
                    (unstableCudaPkgs.${py}.withPackages (pyp: with pyp; [
                        jax
                        jaxlib-bin
                        flax
                        pillow
                        pyarrow
                        datasets
                        wandb
                        matplotlib
                        unstableCudaPkgs.float-tokenization
                        unstableCudaPkgs.ngp-volume-rendering
                        unstableCudaPkgs.serde-helper
                    ]))
                    unstableCudaPkgs.gcc12
                    unstableCudaPkgs.cudaPackages.cudatoolkit
                    unstableCudaPkgs.cudaPackages.cuda_cudart
                    unstableCudaPkgs.cudaPackages.cudnn
                ];
                shellHook = ''
                    export CUDA_PATH=${unstableCudaPkgs.cudatoolkit}
                    source <(sed -Ee '/\$@/d' ${lib.getExe unstableCudaPkgs.nixgl.nixGLIntel})
                    source <(sed -Ee '/\$@/d' ${lib.getExe unstableCudaPkgs.nixgl.auto.nixGLNvidia}*)
                '';
            };
        };
    });
}
