# !/usr/bin/env python

from setuptools import setup, find_packages

with open("README.md") as f:
    long_description = f.read()

setup(
    name="extensible",
    packages=find_packages(".", exclude=["tests"]),
    version="0.1.0",
    install_requires=["jztools", "ploteries", "soleil"],
    author="Joaquin Zepeda",
    description="Context managed hooks",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/zepedaj/extensible",
)
