#!/usr/bin/env python

import setuptools
from distutils.core import setup

setup(name='neurokernel-lamina',
      version='1.0',
      packages=['lamina', 'lamina.vision_models',
                'lamina.neurons', 'lamina.synapses', 'lamina.geometry'],
      install_requires=[
        'configobj >= 5.0.0',
        'neurokernel >= 0.1'
      ]
     )
