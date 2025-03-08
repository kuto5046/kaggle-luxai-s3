import os

from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ext_modules = [
    Pybind11Extension(
        "min_cost_flow",
        ["cpp/MinCostFlow.cpp"],
    ),
]

setup(
    name="min_cost_flow",
    version="0.0.1",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
