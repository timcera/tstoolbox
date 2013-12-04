#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_aggregate
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


class TestAggregate(TestCase):
    def setUp(self):
        dr = pandas.date_range('2011-01-01', periods=2, freq='D')

        ts = pandas.TimeSeries([2.0, 2.0], index=dr)
        self.aggregate_direct_mean = pandas.DataFrame(ts, columns=['Value_mean'])

        ts = pandas.TimeSeries([48.0, 48.0], index=dr)
        self.aggregate_direct_sum = pandas.DataFrame(ts, columns=['Value_sum'])

        self.aggregate_cli_mean = """Datetime,Value_mean
2011-01-01,2.0
2011-01-02,2.0
"""

        self.aggregate_cli_sum = """Datetime,Value_sum
2011-01-01,48.0
2011-01-02,48.0
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
