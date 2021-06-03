# -*- coding: utf-8 -*-
#!/usr/bin/env python

# python setup.py develop --user
# cython distance_matrix.pyx -a

import os
import platform
import numpy as np

try:
  from setuptools import setup
  from setuptools import Extension
  from setuptools import find_packages
  from setuptools import dist
  dist.Distribution().fetch_build_eggs(['Cython>=0.29.0', 'numpy>=1.10'])

except ImportError:
  from distutils.core import setup
  from distutils.core import Extension
  from distutils.core import find_packages

from Cython.Distutils import build_ext
from distutils.sysconfig import customize_compiler



def get_requires (requirements_filename):
  '''
  What packages are required for this module to be executed?
  Parameters
  ----------
    requirements_filename : str
      filename of requirements (e.g requirements.txt)
  Returns
  -------
    requirements : list
      list of required packages
  '''
  with open(requirements_filename, 'r') as fp:
    requirements = fp.read()

  return list(filter(lambda x: x != '', requirements.split()))

def read_description (readme_filename):
  '''
  Description package from filename
  Parameters
  ----------
    readme_filename : str
      filename with readme information (e.g README.md)
  Returns
  -------
    description : str
      str with description
  '''

  try:

    with open(readme_filename, 'r') as fp:
      description = '\n'
      description += fp.read()

    return description

  except IOError:
    return ''

class _build_ext (build_ext):
  '''
  Custom build type
  '''
  def build_extensions (self):
    customize_compiler(self.compiler)
    try:
      self.compiler.compiler_so.remove('-Wstrict-prototypes')
      self.compiler.compiler_so.remove('-Wdate-time')
    except (AttributeError, ValueError):
      pass
    build_ext.build_extensions(self)


here = os.path.abspath(os.path.dirname(__file__))

NAME = 'FemurSegmentation'
DESCRIPTION = ''
URL = ''
EMAIL = ['']
AUTHOR = ['']
REQUIRES_PYTHON = '>=3.5'
VERSION = '0.0.1'
KEYWORDS = ""

CPP_COMPILER = platform.python_compiler()
README_FILENAME = "" #os.path.join(os.getcwd(), 'README.md')
# REQUIREMENTS_FILENAME = os.path.join(here, 'requirements.txt')
# VERSION_FILENAME =

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
  LONG_DESCRIPTION = DESCRIPTION#read_description(README_FILENAME)

except IOError:
  LONG_DESCRIPTION = DESCRIPTION

about = {'__version__' : VERSION}

if 'GCC' in CPP_COMPILER or 'Clang' in CPP_COMPILER:
  cpp_compiler_args = ['-std=c++11', '-std=gnu++11', '-g0']
  compile_args = [ '-Wno-unused-function',
                   '-Wno-narrowing',
                   '-Wall',
                   '-Wextra',
                   '-Wno-unused-result',
                   '-Wno-unknown-pragmas',
                   '-Wfatal-errors',
                   '-Wpedantic',
                   '-march=native',
                   '-Wno-write-strings',
                   '-Wno-overflow',
                   '-Wno-parentheses',
                   '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION'
                 ]
elif 'MSC' in CPP_COMPILER:
  cpp_compiler_args = ['/std:c++latest',
                       '-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION']
  compile_args = []
else:
  raise ValueError('Unknown c++ compiler arg')

whole_compiler_args = sum([cpp_compiler_args, compile_args], [])

ext_modules = [
                Extension(name= '.'.join(['lib', 'GraphCutSupport']),
                          sources=[os.path.join(os.getcwd(), 'maxflow-v3.01', 'graph.cpp'),
                                   os.path.join(os.getcwd(), 'maxflow-v3.01', 'maxflow.cpp'),
                                   os.path.join(os.getcwd(), 'src', 'graphcut.pyx')],
                          libraries=[],
                          library_dirs=[os.path.join('usr', 'lib'),
                                        os.path.join('usr', 'local', 'lib')],
                          include_dirs=[np.get_include(),
                                        os.path.join(os.getcwd(), 'include'),
                                        os.path.join(os.getcwd(), 'maxflow-v3.01')
                                        ],
                          extra_compile_args = whole_compiler_args,
                          language='c++'
                          )
            ]

setup(
        name                          = '{}'.format(NAME),
        version                       = about['__version__'],
        description                   = DESCRIPTION,
        long_description              = LONG_DESCRIPTION,
        long_description_content_type = 'text/markdown',
        author                        = AUTHOR,
        author_email                  = EMAIL,
        maintainer                    = AUTHOR,
        maintainer_email              = EMAIL,
        python_requires               = REQUIRES_PYTHON,
        # install_requires              = get_requires(REQUIREMENTS_FILENAME),
        url                           = URL,
        download_url                  = URL,
        keywords                      = KEYWORDS,
        # packages                      = find_packages(),
        cmdclass                      = {'build_ext': _build_ext},
        license                       = 'MIT',
        ext_modules                   = ext_modules
      )
