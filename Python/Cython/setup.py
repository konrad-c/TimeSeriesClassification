from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy as np

ext_modules=[
    Extension("CDTW",
              ["CDTW.pyx"],
              extra_compile_args = ["/openmp"],
              extra_link_args=['/openmp']
              ) 
]

setup(
    cmdclass = {"build_ext": build_ext},
    ext_modules = ext_modules,
    include_dirs=[np.get_include()]
)