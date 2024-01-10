{
    description = "3D generative AI";

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
        deps = import ./dependencies;
    in flake-utils.lib.eachSystem [ "x86_64-linux" "aarch64-linux" ] (system: let
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
                packageOverrides = finalScope: prevScope: {
                    jax = prevScope.jax.overridePythonAttrs (o: { doCheck = false; });
                    jaxlib = prevScope.jaxlibWithCuda;
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
        cpuPkgs = import nixpkgs {
            inherit system overlays;
            config = {
                allowUnfree = true;
                cudaSupport = false;  # NOTE: disable cuda for cpu env
            };
        };
        mkPythonDeps = { pp, extraPackages }: with pp; [
            pyyaml
            jaxlibWithCuda
            jax
            optax
            flax
            matplotlib
            torch-bin
        ] ++ extraPackages;
        commonShellHook = ''
            [[ "$-" == *i* ]] && exec "$SHELL"
        '';
    in rec {
        default = cudaDevShell;
        cudaDevShell = let  # impure
            isWsl = builtins.pathExists /usr/lib/wsl/lib;
        in cudaPkgs.mkShell {
            name = "cuda";
            buildInputs = [
                (cudaPkgs.${py}.withPackages (pp: mkPythonDeps {
                    inherit pp;
                    extraPackages = with pp; [
                        pkgs.volume-rendering-jax
                        pkgs.jax-tcnn
                        pkgs.safetensors
                    ];
                }))
            ];
            # REF:
            #   <https://github.com/google/jax/issues/5723#issuecomment-1339655621>
            XLA_FLAGS = with builtins; let
                nvidiaDriverVersion =
                    head (match ".*Module  ([0-9\\.]+)  .*" (readFile /proc/driver/nvidia/version));
                nvidiaDriverVersionMajor = lib.toInt (head (splitVersion nvidiaDriverVersion));
            in lib.optionalString
                (!isWsl && nvidiaDriverVersionMajor <= 470)
                "--xla_gpu_force_compilation_parallelism=1";
            shellHook = ''
                source <(sed -Ee '/\$@/d' ${lib.getExe cudaPkgs.nixgl.nixGLIntel})
            '' + (if isWsl
                then ''export LD_LIBRARY_PATH=/usr/lib/wsl/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}''
                else ''source <(sed -Ee '/\$@/d' ${lib.getExe cudaPkgs.nixgl.auto.nixGLNvidia}*)''
            ) + "\n" + commonShellHook;
        };
        cpuDevShell = cpuPkgs.mkShell {
            name = "cpu";
            buildInputs = [
                (cpuPkgs.${py}.withPackages (pp: mkPythonDeps {
                    inherit pp;
                    extraPackages = [];
                }))
            ];
            shellHook = ''
            '' + commonShellHook;
        };
    };
        packages = deps.packages basePkgs;
    }) // {
        overlays.default = deps.overlay;
    };
}
