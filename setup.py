from setuptools import setup, Extension, distutils, Command, find_packages
import setuptools.command.install
import setuptools.command.develop
import setuptools.command.build_py
import distutils.command.build
from shutil import copyfile
import subprocess, os
import glob

TOP_DIR = os.path.realpath(os.path.dirname(__file__))

print('TOP_DIR: ', TOP_DIR)

###############################################################################
# Version and build number
###############################################################################
git_version = subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=TOP_DIR
).decode('ascii').strip()

# TODO (prigoyal): automatically sync this with conda package
tc_version = '0.1.1'
tc_build_number = 2
# versioning should comply with
# https://www.python.org/dev/peps/pep-0440/#public-version-identifiers
# when conda packaging, we get these values from environment variables
if os.getenv('TC_BUILD_VERSION') and os.getenv('TC_BUILD_NUMBER'):
    assert os.getenv('TC_BUILD_NUMBER') is not None, "Please specify valid build number"
    tc_build_number = int(os.getenv('TC_BUILD_NUMBER'))
    tc_version = str(os.getenv('TC_BUILD_VERSION'))
    if tc_build_number > 1:
        tc_version += '.post' + str(tc_build_number)

print('git_version: {} tc_version: {} tc_build_number: {}'.format(
    git_version, tc_version, tc_build_number
))

################################################################################
# Custom override commands
################################################################################
class install(setuptools.command.install.install):
    def run(self):
        setuptools.command.install.install.run(self)

################################################################################
# Extensions
################################################################################
# Extensions built with cmake
ext_modules = []

################################################################################
# Command line options
################################################################################
cmdclass = {
    'install': install,
}

###############################################################################
# Main
###############################################################################
setup(
    name="tensor_comprehensions",
    version=tc_version,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=find_packages(),
    package_data={'tensor_comprehensions': [
        '*.so',
        '*.proto',
    ]},
    install_requires=['pyyaml', 'numpy'],
    author='Tensor Comprehensions Team',
    author_email='tensorcomp@fb.com',
    url='https://github.com/facebookresearch/TensorComprehensions',
    license="Apache 2.0",
    description=("Framework-Agnostic Abstractions for High-Performance Machine Learning"),
)
