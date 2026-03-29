from __future__ import annotations

import os

from setuptools import Extension, setup


extra_compile_args: list[str] = []
if os.name != "nt":
    extra_compile_args.extend(["-O3", "-std=c11"])


setup(
    ext_modules=[
        Extension(
            "ppmdc._cppm",
            ["src/ppmdc/_cppm.c"],
            extra_compile_args=extra_compile_args,
        )
    ]
)
