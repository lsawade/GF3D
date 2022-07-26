""" :noindex:
Setup.py file that governs the installatino process of
`how_to_make_a_python_package` it is used by
`conda install -f environment.yml` which will install the package in an
environment specified in that file.

"""
from setuptools import setup
from setuptools.command.test import test as testcommand

# This installs the pytest command. Meaning that you can simply type pytest
# anywhere and "pytest" will look for all available tests in the current
# directory and subdirectories recursively (not yet supported in the setup.cfg)


class PyTest(testcommand):
    user_options = [('pytest-args=', 'a', "Arguments to pass to py.tests")]

    def initialize_options(self):
        testcommand.initialize_options(self)
        self.pytest_args = []

    def run_tests(self):
        import pytest
        import sys
        errno = pytest.main(self.pytest_args)
        sys.exit(errno)


setup(cmdclass={'tests': PyTest})
