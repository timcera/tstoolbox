#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_tstoolbox
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


class TestDate_slice(TestCase):
    def setUp(self):
        dr = pd.date_range('2011-01-01T12:00:00', periods=3, freq='H')
        self.date_slice = pd.DataFrame([2, 2, 2], index=dr, columns=['Value'])
        self.date_slice_cli = capture(tsutils._printiso, self.date_slice)

    def test_date_slice(self):
        out = tstoolbox.date_slice(input_ts='tests/data_flat.csv', start_date='2011-01-01T12:00:00', end_date='2011-01-01T14:00:00')
        assert_frame_equal(out, self.date_slice)

    def test_date_slice_cli(self):
        args = 'tstoolbox date_slice --input_ts="tests/data_flat.csv" --start_date="2011-01-01T12:00:00" --end_date="2011-01-01T14:00:00"'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate()[0]
        self.assertEqual(out, self.date_slice_cli)
