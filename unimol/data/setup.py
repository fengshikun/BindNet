

from distutils.core import setup, Extension
from Cython.Build import cythonize
import numpy as np

setup(
    name='algos',
    ext_modules=cythonize('algos.pyx'),
    include_dirs=[np.get_include()]
)