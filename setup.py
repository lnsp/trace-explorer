#!/usr/bin/env python

import setuptools

setuptools.setup(name='trace_explorer',
      version='1.0',
      description=['Database trace explorer toolset'],
      author='Lennart Espe',
      author_email='lnsp@users.noreply.github.com',
      url='https://espe.tech/trace-explorer',
      install_requires=['cramjam==2.5.0',
'cycler==0.11.0',
'duckdb==0.3.2 ',
'fastparquet==0.8.0',
'fonttools==4.29.1',
'fsspec==2022.2.0',
'joblib==1.1.0 ',
'kiwisolver==1.3.2',
'matplotlib==3.5.1',
'numpy==1.22.2 ',
'packaging==21.3',
'pandas==1.4.1 ',
'Pillow==9.0.1 ',
'pyparsing==3.0.7',
'python-dateutil==2.8.2',
'pytz==2021.3  ',
'scikit-learn==1.0.2',
'scipy==1.8.0  ',
'six==1.16.0   ',
'sklearn==0.0  ',
'threadpoolctl==3.1.0',
      ],
               
      packages=['trace_explorer'])