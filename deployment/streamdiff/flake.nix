{
    description = "3D character generation.";
    inputs = {
        nixpkgs.url = "github:nixos/nixpkgs/22.11";
        nixpkgs-with-nvidia-driver-fix.url = "github:nixos/nixpkgs/pull/222762/head";
        flake-utils.url = "github:numtide/flake-utils/3db36a8b464d0c4532ba1c7dda728f4576d6d073";
        nixgl = {
            url = "github:guibou/nixgl/c917918ab9ebeee27b0dd657263d3f57ba6bb8ad";
            inputs = {
            nixpkgs.follows = "nixpkgs";
            flake-utils.follows = "flake-utils";
            };
        };
    };
    outputs = inputs@{ self, nixpkgs, flake-utils, ... }: let
        deps = import ./deps;
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
                #opencv4 = prev.opencv4.override {
                #    enableCuda = false;
                #};
                ${py} = prev.${py}.override {
                    packageOverrides = finalScope: prevScope: {
                        jax = prevScope.jax.overridePythonAttrs (o: { doCheck = false; });
                        jaxlib = prevScope.jaxlib-bin;
                        flax = prevScope.flax.overridePythonAttrs (o: {
                            buildInputs = o.buildInputs ++ [ prevScope.pyyaml ];
                            doCheck = false;
                        });
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
                jaxlib-bin
                jax
                optax
                flax
            ] ++ extraPackages;
            commonShellHook = ''
            '';
        in rec {
            default = cudaDevShell;
            cudaDevShell = cudaPkgs.mkShell {
                name = "cuda";
                buildInputs = [
                    (cudaPkgs.${py}.withPackages (pp: mkPythonDeps {
                        inherit pp;
                        extraPackages = with pp; [
                            #pkgs.spherical-harmonics-encoding-jax
                            #pkgs.volume-rendering-jax
                            #pkgs.jax-tcnn
                        ];
                    }))
                ];
                shellHook = ''
                    source <(sed -Ee '/\$@/d' ${lib.getExe cudaPkgs.nixgl.nixGLIntel})
                '' + (if isWsl
                    then ''export LD_LIBRARY_PATH=/usr/lib/wsl/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}''
                    else ''source <(sed -Ee '/\$@/d' ${lib.getExe cudaPkgs.nixgl.auto.nixGLNvidia}*)''
                ) + "\n" + commonShellHook;
            };
        };
        packages = deps.packages basePkgs;
    }) // {
        overlays.default = deps.overlay;
    };
}
