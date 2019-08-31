import os
import sys
import shutil

from setuptools import setup

# temporarily redirect config directory to prevent matplotlib importing
# testing that for writeable directory which results in sandbox error in
# certain easy_install versions
os.environ["MPLCONFIGDIR"] = "."

pkg_name = "tstoolbox"

version = open("VERSION").readline().strip()

if sys.argv[-1] == "publish":
    os.system("python setup.py sdist")

    # The following block of code is to set the timestamp on files to
    # 'now', otherwise ChromeOS/google drive sets to 1970-01-01 and then
    # no one can install it because zip doesn't support dates before
    # 1980.
    os.chdir("dist")
    os.system("tar xvzf {pkg_name}-{version}.tar.gz".format(**locals()))
    os.system("find {pkg_name}-{version}* -exec touch {{}} \\;".format(**locals()))
    os.system(
        "tar czf {pkg_name}-{version}.tar.gz {pkg_name}-{version}".format(**locals())
    )
    shutil.rmtree("{pkg_name}-{version}".format(**locals()))
    os.chdir("..")

    os.system("twine upload dist/{pkg_name}-{version}.tar.gz".format(**locals()))
    sys.exit()

README = open("./README.rst").read()

install_requires = [
    # http://packages.python.org/distribute/setuptools.html#declaring-dependencies
    "dateparser",
    "tabulate",
    "docutils",
    "mando >= 0.4",
    "rst2ansi >= 0.1.5",
    "python-dateutil >= 2.1",
    "numpy",
    "scipy",
    "pandas",
    "pint",
    "matplotlib < 3.1 ; python_version < '3.6'",
    "matplotlib ; python_version >= '3.6'",
    "xlsxwriter",
    "sklearn",
]

setup_requires = []

setup(
    name=pkg_name,
    version=version,
    description="Command line script to manipulate time series files.",
    long_description=README,
    classifiers=[
        # Get strings from
        # http://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Environment :: Console",
        "License :: OSI Approved :: BSD License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    keywords="time_series",
    author="Tim Cera, P.E.",
    author_email="tim@cerazone.net",
    url="http://timcera.bitbucket.io/{pkg_name}/docsrc/index.html".format(**locals()),
    packages=[
        "{pkg_name}".format(**locals()),
        "{pkg_name}.functions".format(**locals()),
        "{pkg_name}.skill_metrics".format(**locals()),
    ],
    include_package_data=True,
    zip_safe=False,
    setup_requires=setup_requires,
    install_requires=install_requires,
    entry_points={
        "console_scripts": ["{pkg_name}={pkg_name}.{pkg_name}:main".format(**locals())]
    },
    test_suite="tests",
    python_requires=">=3.5,!=3.0.*,!=3.1.*,!=3.2.*,!=3.3.*,!=3.4.*",
)
