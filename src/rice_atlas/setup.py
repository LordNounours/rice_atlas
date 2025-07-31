from setuptools import setup, Extension
import pybind11
import numpy

ext_modules = [
    Extension(
        "denoise",
        ["preproc/denoise.cpp"],
        include_dirs=[pybind11.get_include()],
        language="c++",
    ),
]

setup(
    name="functions",
    version="0.1",
    author="Toi",
    description="Suppression de voisins (segmentation prétraitement)",
    ext_modules=ext_modules,
)
