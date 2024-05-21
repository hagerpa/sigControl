from distutils.core import setup

import numpy

setup(name='sigControl',
      version='0.1',
      description='Numerics for the paper "Optimal Control with signatures".',
      url='',
      author='Paul Hager',
      author_email='hagerpa@gmail.com',
      license='MIT',
      packages=['src', 'server'],
      include_dirs=[numpy.get_include()],
      install_requires=['numpy',
                        'iisignature',
                        'matplotlib',
                        'scipy',
                        'pandas',
                        'torch'],
      zip_safe=False)
