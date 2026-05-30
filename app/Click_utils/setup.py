# setup.py
from setuptools import setup, Extension
import pybind11
import sys

ext_modules = [
    Extension(
        'click_utils',                   # name of the resulting module
        ['click_utils.cpp'],             # your C++ source file(s)
        include_dirs=[pybind11.get_include()],
        language='c++',
        extra_compile_args=['/O2'] if sys.platform == 'win32' else ['-O3'],
    ),
]

setup(
    name='click_utils',
    version='1.0',
    ext_modules=ext_modules,
)
