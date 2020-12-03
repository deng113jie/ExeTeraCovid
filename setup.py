from setuptools import setup

from exeteracovid import __version__

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='exeteracovid',
    version=__version__,
    description='Analytics for the Covid Symptom Study, based on ExeTera',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/kcl-bmeis/ExeTeraCovid',
    author='Ben Murray',
    author_email='benjamin.murray@kcl.ac.uk',
    license='http://www.apache.org/licenses/LICENSE-2.0',
    packages=['exeteracovid', 'exeteracovid.algorithms', 'exeteracovid.processing'],
    install_requires=[
        'exetera>=0.4'
    ]
)
