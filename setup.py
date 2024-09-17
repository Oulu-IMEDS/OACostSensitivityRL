from setuptools import setup, find_packages

setup(
    name='OACostRL',
    version='0.1',
    author='Khanh Nguyen',
    author_email='khanh.nguyen@oulu.fi',
    packages=find_packages(),
    include_package_data=True,
    license='LICENSE.txt',
    long_description=open('README.md').read(),
)