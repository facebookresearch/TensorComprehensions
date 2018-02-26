from setuptools import setup, Extension, distutils, Command, find_packages
import setuptools.command.install
import setuptools.command.develop
import setuptools.command.build_py
import distutils.command.build
from shutil import copyfile
import subprocess, os

TOP_DIR = os.path.realpath(os.path.dirname(__file__))

print('TOP_DIR: ', TOP_DIR)

###############################################################################
# Version and build number
###############################################################################

git_version = subprocess.check_output(
    ['git', 'rev-parse', 'HEAD'], cwd=TOP_DIR
).decode('ascii').strip()

tc_version = '0.1.0'
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
# Copy the proto files only
################################################################################
copyfile("src/proto/compcache.proto", "tensor_comprehensions/compilation_cache.proto")
copyfile("src/proto/mapping_options.proto", "tensor_comprehensions/mapping_options.proto")

################################################################################
# Custom override commands
################################################################################

class build_py(setuptools.command.build_py.build_py):
    def run(self):
        self.create_version_file()
        setuptools.command.build_py.build_py.run(self)

    @staticmethod
    def create_version_file():
        global tc_version, tc_build_number, git_version
        print('BUILD git_version: {} tc_version: {} tc_build_number: {}'.format(
            git_version, tc_version, tc_build_number
        ))
        with open(
            os.path.join(TOP_DIR, 'tensor_comprehensions', 'version.py'), 'w'
        ) as fopen:
            fopen.write("__version__ = '{}'\n".format(str(tc_version)))
            fopen.write("build_number = {}\n".format(int(tc_build_number)))
            fopen.write("git_version = '{}'\n".format(str(git_version)))
        print('Version file written.')


class install(setuptools.command.install.install):
    def run(self):
        setuptools.command.install.install.run(self)


class develop(setuptools.command.develop.develop):
    def run(self):
        build_py.create_version_file()
        setuptools.command.develop.develop.run(self)

################################################################################
# Extensions
################################################################################

# we have build extensions in the cmake command so no extensions left to build
print('All extension module were build with cmake')
ext_modules = []

################################################################################
# Command line options
################################################################################

cmdclass = {
    'develop': develop,
    'build_py': build_py,
    'install': install,
}

###############################################################################
# Pure python packages
###############################################################################
# don't include the tests in the conda package, we run these tests when building
# the package
packages = find_packages(exclude=('test_python', 'test_python.*'))

###############################################################################
# Main
###############################################################################


setup(
    name="tensor_comprehensions",
    version=tc_version,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    packages=packages,
    package_data={'tensor_comprehensions': [
        '*.so',
        '*.proto',
        'library/*.yaml',
    ]},
    install_requires=['pyyaml', 'numpy'],
    author='prigoyal',
    author_email='prigoyal@fb.com',
    url='https://github.com/facebookresearch/TensorComprehensions',
    license="Apache 2.0",
    description=("Framework-Agnostic Abstractions for High-Performance Machine Learning"),
)
