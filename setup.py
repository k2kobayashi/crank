#!/usr/bin/env python

import os

from setuptools import find_packages, setup

if os.path.exists("README.md"):
    README = open("README.md").read()
else:
    README = ""  # a placeholder, readme is generated on release
CHANGES = open("CHANGES.md").read()

setup(
    name="crank-vc",
    version="0.4.1",
    description="Non-parallel Voice Conversion called crank",
    url="https://github.com/k2kobayashi/crank",
    author="K. KOBAYASHI",
    packages=find_packages(exclude=["egs", "test", "utils"]),
    long_description=(README + "\n" + CHANGES),
    long_description_content_type='text/markdown',
    license="MIT",
    install_requires=open("tools/requirements.txt").readlines(),
)
