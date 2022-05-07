#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script only setups the 
environment to be used with the tandem class"""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

# with open('HISTORY.rst') as history_file:
    # history = history_file.read()

requirements = ['numpy',  'matplotlib', 'scipy', 'sklearn', 'datetime']


setup(
    name='tandems',
    version='0.989',
    description="""Calculate yearly average efficiencies for multijunction tandem solar cells""",
    long_description=readme + '\n\n',# + history,
    author="Jose M. Ripalda",
    author_email='j.ripalda@csic.es',
    url='https://github.com/Ripalda/Tandems',
    packages=find_packages(include=['tandems']),
    include_package_data=True,
    install_requires=requirements,
    license="GNU General Public License (GPL)",
    zip_safe=False,
    keywords='tandems',
    classifiers=[
        'Operating System :: OS Independent',
        'Intended Audience :: Science',
        'License :: OSI Approved :: GNU General Public License (GPL)',
        'Natural Language :: English',
        "Programming Language :: Python :: 3"
    ],
    #data_files=[('example_data', ['tandems/data/'])],
    # test_suite='tests',
    # tests_require=test_requirements,
    # setup_requires=setup_requirements,
)
