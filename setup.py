#!/usr/bin/env python

from setuptools import setup, find_packages

import os
if os.path.exists('../README.md'):
    README = open('../README.md').read()
else:
    README = ""  # a placeholder, readme is generated on release
CHANGES = open('CHANGES.md').read()

setup(
    name="crank",
    version="0.20",
    description="Non-parallel Voice Conversion",
    url='https://github.com/k2kobayashi/crank',
    author='K. KOBAYASHI',
    packages=find_packages(exclude=['shell, stage']),
    long_description=(README + '\n' + CHANGES),
    license='MIT',
    install_requires=open('tools/requirements.txt').readlines(),
)
