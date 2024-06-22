{
    description = "3D generative AI";

    inputs = {
        nixpkgs-unstable.url = "github:nixos/nixpkgs/nixpkgs-unstable";
        flake-utils.url = "github:numtide/flake-utils";
        # Patched version of nixGL from kenrandunderscore.
        # PR: https://github.com/nix-community/nixGL/pull/165
        # TODO: switch back to github:nix-community/nixGL when PR is merged.
        nixgl.url = "github:hayden-donnelly/nixGL";
    };
    outputs = inputs@{ 
        self, 
        nixpkgs-unstable, 
        flake-utils, 
        nixgl, 
        ... 
    }: flake-utils.lib.eachSystem [ "x86_64-linux" ] (system: 
        let
            inherit (nixpkgs-unstable) lib;
            pyVer = "311";
            py = "python${pyVer}";
            submodules = import ./submodules;
            overlays = [
                submodules.overlay
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
                            # For some reason flax has jaxlib as a buildInput and tensorflow as a 
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
                                pythonImportsCheck = [];
                                doCheck = false;
                            });
                            safetensors = prevPkgs.safetensors.overridePythonAttrs(o: {
                                # Remove torch from nativeCheckInputs.
                                nativeCheckInputs = with prevPkgs; [ h5py numpy pytestCheckHook ];
                                doCheck = false; 
                            });
                        };
                    };
                    glad = ./external/glad { inherit prev; };
                })
            ];
            pkgs = import nixpkgs-unstable {
                inherit system overlays;
                config = {
                    allowUnfree = true;
                    cudaSupport = true;
                };
            };
        in {
            devShells = {
                default = pkgs.mkShell {
                    name = "cuda";
                    buildInputs = with pkgs; [
                        (pkgs.${py}.withPackages (pyp: with pyp; [
                            jax
                            jaxlib-bin
                            flax
                            pillow
                            pyarrow
                            datasets
                            wandb
                            matplotlib
                            pytest
                            safetensors
                            ngp-volume-rendering
                        ]))
                    ];
                    shellHook = ''
                        export CUDA_PATH=${pkgs.cudatoolkit}
                        source <(sed -Ee '/\$@/d' ${lib.getExe pkgs.nixgl.nixGLIntel})
                        source <(sed -Ee '/\$@/d' ${lib.getExe pkgs.nixgl.auto.nixGLNvidia}*)
                    '';
                };
            };
        }
    );
}
