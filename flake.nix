{
    description = "3D character generation.";
    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/23.11";
        nixpkgs-with-nvidia-driver-fix.url = "github:nixos/nixpkgs/pull/222762/head";
        flake-utils.url = "github:numtide/flake-utils";
        nixgl = {
            url = "github:guibou/nixgl";
            inputs = {
                nixpkgs.follows = "nixpkgs";
                flake-utils.follows = "flake-utils";
            };
        };
    };
    outputs = inputs@{ self, nixpkgs, flake-utils, ... }: let
        deps = import ./dependencies;
    in flake-utils.lib.eachSystem [ "x86_64-linux" ] (system: let
        inherit (nixpkgs) lib;
        basePkgs = import nixpkgs {
            inherit system;
            overlays = [
                self.overlays.default
            ];
        };
    in {
        devShells = let
            pyVer = "310";
            py = "python${pyVer}";
            jaxOverlays = final: prev: {
                ${py} = prev.${py}.override {
                    packageOverrides = final2: prev2: {
                        # Turn off jax import check so it doesn't fail due to jaxlib not being installed.
                        jax = prev2.jax.overridePythonAttrs (o: { pythonImportsCheck = []; doCheck = false; });
                        # Use jaxlib-bin instead of jaxlib because it takes a long time to build XLA.
                        jaxlib = prev2.jaxlib-bin;
                        flax = prev2.flax.overridePythonAttrs (o: { doCheck = false; });
                    };
                };
            };
            overlays = [
                inputs.nixgl.overlays.default
                self.overlays.default
                jaxOverlays
            ];
            cudaPkgs = import nixpkgs {
                inherit system overlays;
                config = {
                    allowUnfree = true;
                    cudaSupport = true;
                    packageOverrides = pkgs: {
                        linuxPackages = (import inputs.nixpkgs-with-nvidia-driver-fix {}).linuxPackages;
                    };
                };
            };
            mkPythonDeps = { pp, extraPackages }: with pp; [
                pyyaml
                jaxlib-bin
                jax
                optax
                flax
            ] ++ extraPackages;
            commonShellHook = '''';
        in rec {
            default = cudaDevShell;
            cudaDevShell = cudaPkgs.mkShell {
                name = "cuda";
                buildInputs = [
                    (cudaPkgs.${py}.withPackages (pp: mkPythonDeps {
                        inherit pp;
                        extraPackages = with pp; [
                            pkgs.volume-rendering-jax
                            pkgs.jax-tcnn
                        ];
                    }))
                ];
                shellHook = ''
                    source <(sed -Ee '/\$@/d' ${lib.getExe cudaPkgs.nixgl.nixGLIntel})
                    source <(sed -Ee '/\$@/d' ${lib.getExe cudaPkgs.nixgl.auto.nixGLNvidia}*)
                '' + "\n" + commonShellHook;
            };
        };
        packages = deps.packages basePkgs;
    }) // {
        overlays.default = deps.overlay;
    };
}