from setuptools import setup
from setuptools import find_packages

setup(name='reid',
      version='1.0',
      description='Person Re-identification in PyTorch',
      author='Hoang-Quan Nguyen',
      author_email='hoangquan.qti@gmail.com',
      install_requires=['numpy>=1.19.5',
                        'torch>=1.5.0',
                        'scipy>=1.4.1'
                        ],
      packages=find_packages())
