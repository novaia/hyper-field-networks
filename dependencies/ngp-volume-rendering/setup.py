#!/usr/bin/env python

import os
import subprocess

from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

HERE = os.path.dirname(os.path.realpath(__file__))


class CMakeBuildExt(build_ext):
    def build_extensions(self):
        # First: configure CMake build
        import platform
        import sys
        import sysconfig

        import pybind11

        # Work out the relevant Python paths to pass to CMake, adapted from the
        # PyTorch build system
        if platform.system() == "Windows":
            cmake_python_library = "{}/libs/python{}.lib".format(
                sysconfig.get_config_var("prefix"),
                sysconfig.get_config_var("VERSION"),
            )
            if not os.path.exists(cmake_python_library):
                cmake_python_library = "{}/libs/python{}.lib".format(
                    sys.base_prefix,
                    sysconfig.get_config_var("VERSION"),
                )
        else:
            cmake_python_library = "{}/{}".format(
                sysconfig.get_config_var("LIBDIR"),
                sysconfig.get_config_var("INSTSONAME"),
            )
        cmake_python_include_dir = sysconfig.get_path("include")

        install_dir = os.path.abspath(
            os.path.dirname(self.get_ext_fullpath("dummy"))
        )
        os.makedirs(install_dir, exist_ok=True)
        cmake_args = [
            "-DCMAKE_INSTALL_PREFIX={}".format(install_dir),
            "-DCMAKE_CUDA_STANDARD=17",
            "-DPython_EXECUTABLE={}".format(sys.executable),
            "-DPython_LIBRARIES={}".format(cmake_python_library),
            "-DPython_INCLUDE_DIRS={}".format(cmake_python_include_dir),
            "-DCMAKE_BUILD_TYPE={}".format(
                "Debug" if self.debug else "Release"
            ),
            "-DCMAKE_PREFIX_PATH={}".format(pybind11.get_cmake_dir()),
            "-G Unix Makefiles",
        ]
        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(
            ["cmake", HERE] + cmake_args, cwd=self.build_temp
        )

        # Build all the extensions
        super().build_extensions()

        # Finally run install
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", "install"],
            cwd=self.build_temp,
        )

    def build_extension(self, ext):
        target_name = ext.name.split(".")[-1]
        subprocess.check_call(
            ["cmake", "--build", ".", "--target", target_name],
            cwd=self.build_temp,
        )

extensions = [
    Extension(
        # Python dotted name, whose final component should be a buildable target defined in CMakeLists.txt.
        "ngp_volume_rendering.cuda_ffi",
        [  
            # Source paths, relative to this setup.py file.
            "csrc/ffi.cc",
            "csrc/packbits.cu",
            "csrc/marching.cu",
            "csrc/integrating.cu",
        ],
    ),
]

setup(
    name="ngp-volume-rendering",
    author="blurgyy",
    license="Apache License, Version 2.0",
    package_dir={"": "pysrc"},
    packages=find_packages("pysrc"),
    include_package_data=True,
    install_requires=["jax", "jaxlib", "chex"],
    ext_modules=extensions,
    cmdclass={"build_ext": CMakeBuildExt},
)
