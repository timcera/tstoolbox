#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_tstoolbox
----------------------------------

Tests for `tstoolbox` module.
"""

from pandas.util.testing import TestCase
from pandas.util.testing import assert_frame_equal
import sys
import subprocess
import shlex
try:
    from cStringIO import StringIO
except:
    from io import StringIO

import pandas

from tstoolbox import tstoolbox


def capture(func, *args, **kwds):
    sys.stdout = StringIO()      # capture output
    out = func(*args, **kwds)
    out = sys.stdout.getvalue()  # release output
    return out


class TestConvert(TestCase):
    def setUp(self):
        dr = pandas.date_range('2000-01-01', periods=2, freq='D')
        ts = pandas.TimeSeries([4.5, 4.6], index=dr)
        self.compare_direct_01 = pandas.DataFrame(ts, columns=['Value_convert'])
        self.compare_direct_01.index.name = 'Datetime'

        dr = pandas.date_range('2000-01-01', periods=2, freq='D')
        ts = pandas.TimeSeries([11.0, 11.2], index=dr)
        self.compare_direct_02 = pandas.DataFrame(ts, columns=['Value_convert'])
        self.compare_direct_02.index.name = 'Datetime'

        self.compare_cli_01 = b"""Datetime,Value_convert
2000-01-01,4.5
2000-01-02,4.6
"""
        self.compare_cli_02 = b"""Datetime,Value_convert
2000-01-01,11
2000-01-02,11.2
"""

    def test_convert_direct_01(self):
        out = tstoolbox.convert(input_ts='tests/test.csv')
        assert_frame_equal(out, self.compare_direct_01)

    def test_convert_direct_02(self):
        out = tstoolbox.convert(input_ts='tests/test.csv', factor=2, offset=2)
        assert_frame_equal(out, self.compare_direct_02)

    def test_convert_cli_01(self):
        args = 'tstoolbox convert --input_ts="tests/test.csv"'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.compare_cli_01)

    def test_convert_cli_02(self):
        args = 'tstoolbox convert --factor=2 --offset=2 --input_ts="tests/test.csv"'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.compare_cli_02)
