#!/usr/bin/env python
#-*- coding:utf-8 -*-
##
## setup.py
##
##  Created on: Jan 23, 2018
##      Author: Alexey S. Ignatiev
##      E-mail: aignatiev@ciencias.ulisboa.pt
##

#
#==============================================================================
import distutils.command.build
import distutils.command.install
from distutils.command.clean import clean
from distutils.core import setup, Extension
import numpy.distutils.misc_util

import inspect, os, sys
sys.path.insert(0, os.path.join(os.path.realpath(os.path.abspath(os.path.split(inspect.getfile(inspect.currentframe()))[0])), 'solvers/'))
import prepare

from pysat import __version__

#
#==============================================================================
ROOT = os.path.abspath(os.path.dirname(__file__))

LONG_DESCRIPTION = """
A Python library providing a simple interface to a number of
state-of-art Boolean satisfiability (SAT) solvers and a few types of
cardinality encodings. The purpose of PySAT is to enable researchers
working on SAT and its applications and generalizations to easily
prototype with SAT oracles in Python while exploiting incrementally the
power of the original low-level implementations of modern SAT solvers.

With PySAT it should be easy for you to implement a MaxSAT solver, an
MUS/MCS extractor/enumerator, or any tool solving an application problem
with the (potentially multiple) use of a SAT oracle.

Details can be found at `https://pysathq.github.io <https://pysathq.github.io>`__.
"""

# solvers to install
#==============================================================================
# to_install = ['glucose30', 'glucose41', 'lingeling', 'minicard', 'minisat22',
#         'minisatgh']

to_install = ['sharpSAT', 'glucose30']
# example scripts to install as standalone executables
#==============================================================================
scripts = ['fm', 'genhard', 'lbx', 'lsu', 'mcsls', 'musx', 'rc2']


# we need to redefine the build command to
# be able to download and compile solvers
#==============================================================================
class build(distutils.command.build.build):
    """
        Our custom builder class.
    """

    def run(self):
        """
            Download, patch and compile SAT solvers before building.
        """

        # download and compile solvers
        prepare.do(to_install)

        # now, do standard build
        distutils.command.build.build.run(self)

# class clean(distutils.command.clean.clean):

#     def run(self):
#         # Clean all solvers
#         distutils.command.clean.clean.run(self)


# C extensions: pycard and pysolvers
#==============================================================================
pycard_ext = Extension('pycard',
    sources=['cardenc/pycard.cc'],
    extra_compile_args=['-std=c++11', '-Wno-deprecated'],#, '-stdlib=libc++'],
    include_dirs=['cardenc'] ,
    language='c++',
    libraries=['stdc++'],
    library_dirs=[]
)
 # '-stdlib=libc++'] + \
pysolvers_ext = Extension('pysolvers',
    sources = ['solvers/pysolvers.cc'],
    extra_compile_args = ['-std=c++11', '-Wno-deprecated'] + \
        list(map(lambda x: '-DWITH_{0}'.format(x.upper()), to_install)),
    include_dirs = ['solvers'],
    language = 'c++',
    libraries = to_install + ['stdc++', 'gmp', 'gmpxx'],
    library_dirs = list(map(lambda x: os.path.join('solvers', x), to_install))
)


# finally, calling standard distutils.core.setup()
#==============================================================================
setup(name='python-sat',
    packages=['pysat', 'pysat.examples'],
    package_dir={'pysat.examples': 'examples'},
    version=__version__,
    description='A Python library for prototyping with SAT oracles',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/x-rst; charset=UTF-8',
    license='MIT',
    author='Alexey Ignatiev, Joao Marques-Silva, Antonio Morgado',
    author_email='aignatiev@ciencias.ulisboa.pt, jpms@ciencias.ulisboa.pt, ajmorgado@ciencias.ulisboa.pt',
    url='https://github.com/pysathq/pysat',
    ext_modules=[pycard_ext, pysolvers_ext],
    scripts=['examples/{0}.py'.format(s) for s in scripts],
    cmdclass={'build': build},#, 'clean': clean},
    install_requires=['six'],
    include_dirs=numpy.distutils.misc_util.get_numpy_include_dirs(),
)
