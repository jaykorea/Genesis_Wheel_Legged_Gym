from setuptools import find_packages
from distutils.core import setup

setup(
    name='genesis_wlr',
    version='0.1.0',
    author='Yasen Jia',
    license="BSD-3-Clause",
    packages=find_packages(),
    author_email='jh.cho@cocelo.ai',
    description='Genesis environments for Legged-wheeled Robots',
    install_requires=['matplotlib',
                      'pymeshlab',
                      'prettytable',
                      'rsl-rl-lib',
                      'onnxruntime',]
)

