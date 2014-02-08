#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_describe
----------------------------------

Tests for `tstoolbox` module.
"""

import shlex
import subprocess
from pandas.util.testing import TestCase
from pandas.util.testing import assert_frame_equal
import sys
try:
    from cStringIO import StringIO
except:
    from io import StringIO

import pandas as pd

from tstoolbox import tstoolbox
import tstoolbox.tsutils as tsutils


def capture(func, *args, **kwds):
    sys.stdout = StringIO()      # capture output
    out = func(*args, **kwds)
    out = sys.stdout.getvalue()  # release output
    try:
        out = bytes(out, 'utf-8')
    except:
        pass
    return out


class TestDescribe(TestCase):
    def setUp(self):
        index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        self.date_slice = pd.DataFrame([1672.000000, 836.905383,
            843.901292, 0.000000, 158.425000, 578.800000, 1253.450000,
            4902.000000], index=index,
                columns=['Area'])
        self.date_slice_cli = capture(tsutils._printiso, self.date_slice)

    def test_date_slice(self):
        out = tstoolbox.describe(input_ts='tests/sunspot_area.csv')
        assert_frame_equal(out, self.date_slice)

    def test_date_slice_cli(self):
        args = 'tstoolbox describe --input_ts="tests/sunspot_area.csv"'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate()[0]
        self.assertEqual(out, self.date_slice_cli)
