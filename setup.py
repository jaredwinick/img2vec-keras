from setuptools import setup

setup(name='img2vec_keras',
      version='0.2',
      description='Image to dense vector embedding',
      url='https://github.com/jaredwinick/img2vec-keras',
      author='Jared Winick',
      author_email='jaredwinick@gmail.com',
      license='Apache License 2.0',
      packages=['img2vec_keras'],
      zip_safe=False,
      install_requires=[
          'numpy>=1.9.1',
          'tensorflow>=2.0.0',
          'pillow'
      ])