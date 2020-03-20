from setuptools import find_packages, setup

try: # for pip >= 10
    from pip._internal.req import parse_requirements
except ImportError: # for pip <= 9.0.3
    from pip.req import parse_requirements

import os
import sys
import subprocess

# Utility function to read the README file.
# Used for the long_description.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

# parse_requirements() returns generator of pip.req.InstallRequirement objects
install_reqs = parse_requirements('requirements/requirements-py3.txt', session='hack')

reqs = [str(ir.req) for ir in install_reqs]


with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='UniRep',
    version='0.0.1',
    description='UniRep',
    author='churchlab',
    project_urls={
        'Documentation': 'https://readthedocs.org', # TODO
        'Source': 'https://github.com/akmorrow13/UniRep'
    },
    classifiers=[
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        # Pick your license as you wish (should match "license" above)
         'License :: OSI Approved :: Apache Software License',
        # Python versions supported
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8'
    ],
    license="GNU GENERAL PUBLIC LICENSE",
    keywords='protein engineering',
    install_requires=reqs,
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['*.test.*']),
    python_requires='>=3'
)
 
