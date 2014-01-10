#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_read
----------------------------------

Tests for `tstoolbox` module.
"""

from unittest import TestCase
import sys
import shlex
import subprocess
try:
    from cStringIO import StringIO
except:
    from io import StringIO

import pandas

import tstoolbox


def capture(func, *args, **kwds):
    sys.stdout = StringIO()      # capture output
    out = func(*args, **kwds)
    out = sys.stdout.getvalue()  # release output
    return out


class TestRead(TestCase):
    def setUp(self):
        dr = pandas.date_range('2000-01-01', periods=2, freq='D')

        ts = pandas.TimeSeries([4.5, 4.6], index=dr)

        self.read_direct = pandas.DataFrame(ts, columns=['test_Value'])

        self.read_multiple_direct = pandas.DataFrame(ts, columns=['test_Value'])
        self.read_multiple_direct = self.read_multiple_direct.join(pandas.Series(ts, name='test_1_Value'))

        self.read_cli = b"""Datetime,test_Value
2000-01-01,4.5
2000-01-02,4.6
"""

        self.read_multiple_cli = b"""Datetime,test_Value,test_1_Value
2000-01-01,4.5,4.5
2000-01-02,4.6,4.6
"""

    def test_read_direct(self):
        out = tstoolbox.read('tests/test.csv')
        self.assertEqual(out, self.read_direct)

    def test_read_mulitple_direct(self):
        out = tstoolbox.read('tests/test.csv,tests/test.csv')
        self.assertEqual(out, self.read_multiple_direct)

    def test_read_cli(self):
        args = 'tstoolbox read tests/test.csv'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.read_cli)

    def test_read_mulitple_cli(self):
        args = 'tstoolbox read tests/test.csv,tests/test.csv'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.read_multiple_cli)
