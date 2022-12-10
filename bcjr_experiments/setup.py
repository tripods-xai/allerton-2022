from setuptools import find_packages, setup

setup(
    name="src",
    packages=find_packages(include=["src", "old_src"]),
    version="0.1.0",
    description="research source code",
    author="abhmul",
    license="MIT",
)
