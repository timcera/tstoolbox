# -*- coding: utf-8 -*-

import os
import shlex
import subprocess
import sys

from setuptools import setup

# temporarily redirect config directory to prevent matplotlib importing
# testing that for writeable directory which results in sandbox error in
# certain easy_install versions
os.environ["MPLCONFIGDIR"] = "."

pkg_name = "tstoolbox"

version = open("VERSION").readline().strip()

if sys.argv[-1] == "publish":
    subprocess.run(shlex.split("cleanpy ."), check=True)
    subprocess.run(shlex.split("python setup.py sdist"), check=True)
    subprocess.run(
        shlex.split(f"twine upload dist/{pkg_name}-{version}.tar.gz"), check=True
    )
    sys.exit()

setup(
    test_suite="tests",
)
