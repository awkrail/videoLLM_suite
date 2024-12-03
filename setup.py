from setuptools import setup, find_packages

setup(
    name='firefly',
    version='0.1',
    install_requires=['discord'],
    packages=find_packages(exclude=['demo']),
)