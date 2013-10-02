
import sys
import os

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

if sys.argv[-1] == 'publish':
    os.system('python setup.py sdist upload')
    sys.exit()

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.rst')).read()
CHANGES = open(os.path.join(here, 'CHANGES.rst')).read()

version = open("VERSION").readline().strip()

install_requires = [
    # List your project dependencies here.
    # For more details, see:
    # http://packages.python.org/distribute/setuptools.html#declaring-dependencies
    'pandas >= 0.8.1',
    'baker >= 1.3',
    'scipy',
]


setup(name='tstoolbox',
      version=version,
      description="Command line script to manipulate time series files.",
      long_description=README + '\n\n' + CHANGES,
      classifiers=[
      # Get strings from http://pypi.python.org/pypi?%3Aaction=list_classifiers
          'Development Status :: 4 - Beta',
          'Intended Audience :: Science/Research',
          'Intended Audience :: End Users/Desktop',
          'Environment :: Console',
          'License :: OSI Approved :: GNU General Public License (GPL)',
          'Natural Language :: English',
          'Operating System :: OS Independent',
          'Programming Language :: Python :: 2',
          'Topic :: Scientific/Engineering',
      ],
      keywords='time_series',
      author='Tim Cera, P.E.',
      author_email='tim@cerazone.net',
      url='http://pypi.python.org/pypi/tstoolbox',
      license='GPL2',
      packages=['tstoolbox', 'tsutils'],
      include_package_data=True,
      zip_safe=False,
      install_requires=install_requires,
      entry_points={
          'console_scripts':
              ['tstoolbox=tstoolbox:main']
      },
      test_suite='tests',
      )
