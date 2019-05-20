# encoding: utf-8
from setuptools import setup

setup(name='talkie',
      version='0.4',
      description='Visually grounded speech models in Pytorch',
      url='https://github.com/gchrupala/talkie',
      author='Grzegorz Chrupała',
      author_email='g.chrupala@uvt.nl',
      license='MIT',
      packages=['onion','vg', 'vg.defn'],
      zip_safe=False,
      install_requires=[
          'numpy == 1.16.2',
          'torch >= 1.0.1.post2',
          'torchvision >= 0.2.2.post3',
          'scikit-learn == 0.20.3',
          'SoundFile == 0.10.2',
          'python-speech-features == 0.6',
          'python-Levenshtein == 0.12.0'

      ])
