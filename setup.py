#!/usr/bin/python3

import setuptools

long_description = \
"This project was originally implemented by " \
"@dudley-fitzgerald as a collection of a seismic " \
"calculation. This specific one enhances and " \
"includes new attributes, other facilities and " \
"some optimizations to help geologists to " \
"to calculate them."

setuptools.setup(
    name="d2geo",
    version="1.0.0",
    author="Julio Cesar Faracco",
    author_email="jcfaracco@gmail.com",
    description="A collections of tools to calculate seismic attributes",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jcfaracco/d2geo",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache License, Version 2.0",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(where="."),
    python_requires=">=3.6",
)
