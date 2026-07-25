from setuptools import setup, find_packages

setup(name='badgyal',
      version='0.0.7',
      description='pytorch badgyal and meangirl inference engine',
      author='dkappe',
      url='https://github.com/dkappe/badgyal',
      packages=find_packages(),
      package_data={'badgyal': ['*.pb.gz', '*.pt']},
      install_requires=[
          'numpy==1.23.5',
          'protobuf==3.12.4',
          'torch==1.13.1',
          'pylru==1.2.1',
          'chess'
      ]
)
