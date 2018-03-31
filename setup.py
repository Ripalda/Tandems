#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script only setups the 
environment to be used with the tandem class"""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

# with open('HISTORY.rst') as history_file:
    # history = history_file.read()

requirements = ['numpy', 'scipy', 'matplotlib','json_tricks', 
]


setup(
    name='Tandem',
    version='1',
    description="""Machine Learning for realistic yearly averaged photovoltaic
    efficiency calculations""",
    long_description=readme + '\n\n',# + history,
    author="J.M. ipalda",
    author_email='ripalda@imm.cnm.csic.es',
    url='https://github.com/Ripalda/Tandems',
    packages=find_packages(include=['tandems']),
    include_package_data=True,
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    keywords='tandems',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
    ],
    # test_suite='tests',
    # tests_require=test_requirements,
    # setup_requires=setup_requirements,
)
