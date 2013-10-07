#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_tstoolbox
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

        self.read_direct = pandas.DataFrame(ts, columns=['Value'])

        self.read_multiple_direct = pandas.DataFrame(ts, columns=['test_Value'])
        self.read_multiple_direct = self.read_multiple_direct.join(pandas.Series(ts, name='test_1_Value'))

        self.read_cli = b"""Datetime, Value
2000-01-01 00:00:00 ,  4.5
2000-01-02 00:00:00 ,  4.6
"""

        self.read_multiple_cli = b"""Datetime, test_Value, test_1_Value
2000-01-01 00:00:00 ,  4.5, 4.5
2000-01-02 00:00:00 ,  4.6, 4.6
"""

    def test_read_direct(self):
        out = tstoolbox.read('tests/test.csv')
        self.assertEqual(out, self.read_direct)

    def test_read_mulitple_direct(self):
        out = tstoolbox.read('tests/test.csv', 'tests/test.csv')
        self.assertEqual(out, self.read_multiple_direct)

    def test_read_cli(self):
        args = 'tstoolbox read "tests/test.csv"'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.read_cli)

    def test_read_mulitple_cli(self):
        args = 'tstoolbox read tests/test.csv tests/test.csv'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.read_multiple_cli)


class TestAggregate(TestCase):
    def setUp(self):
        dr = pandas.date_range('2011-01-01', periods=2, freq='D')

        ts = pandas.TimeSeries([2.0, 2.0], index=dr)
        self.aggregate_direct_mean = pandas.DataFrame(ts, columns=['Value_mean'])

        ts = pandas.TimeSeries([48.0, 48.0], index=dr)
        self.aggregate_direct_sum = pandas.DataFrame(ts, columns=['Value_sum'])

        self.aggregate_cli_mean = """Datetime, Value_mean
2011-01-01 00:00:00 ,  2.0
2011-01-02 00:00:00 ,  2.0
"""

        self.aggregate_cli_sum = """Datetime, Value_sum
2011-01-01 00:00:00 ,  48.0
2011-01-02 00:00:00 ,  48.0
"""

    def test_aggregate_direct_mean(self):
        out = tstoolbox.aggregate(statistic='mean', agg_interval='daily', input_ts='tests/test_aggregate.csv')
        self.assertEqual(out, self.aggregate_direct_mean)

    def test_aggregate_direct_sum(self):
        out = tstoolbox.aggregate(statistic='sum', agg_interval='daily', input_ts='tests/test_aggregate.csv')
        self.assertEqual(out, self.aggregate_direct_sum)

    def test_aggregate_cli_mean(self):
        args = 'tstoolbox aggregate --statistic="mean" --input_ts="tests/test_aggregate.csv"'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out, self.aggregate_direct_mean)

    def test_aggregate_cli_sum(self):
        args = 'tstoolbox aggregate --statistic="sum" --input_ts="tests/test_aggregate.csv"'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out, self.aggregate_direct_sum)
