#!/usr/bin/env python

from setuptools import setup, Extension
from torch.utils import cpp_extension
from Cython.Build import cythonize


mise_module = Extension(
    'utils.mise',
    sources=['utils/mise.pyx'],
)

# group_linear = cpp_extension.CppExtension(
#     name='model.group_linear', 
#     sources=['src/group_linear/group_linear.cpp'],
# )

setup(name='UNIF',
      version='1.0',
      description='The official implementation of UNIF (United Neural Implicit Functions).',
      author='Shenhan Qian',
      author_email='qianshh@shanghaitech.edu.cn',
    #   url='',
      packages=['.'],
      ext_modules=cythonize(mise_module),
      # cmdclass={'build_ext': cpp_extension.BuildExtension}
)
