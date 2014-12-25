#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_aggregate
----------------------------------

Tests for `tstoolbox` module.
"""

from pandas.util.testing import TestCase
from pandas.util.testing import assert_frame_equal
import sys
import shlex
import subprocess
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


class TestAggregate(TestCase):
    def setUp(self):
        dr = pandas.date_range('2011-01-01', periods=2, freq='D')

        ts = pandas.TimeSeries([2, 2], index=dr)
        self.aggregate_direct_mean = pandas.DataFrame(ts,
                                                      columns=['Value_mean'])
        self.aggregate_direct_mean.index.name = 'Datetime'

        ts = pandas.TimeSeries([48, 48], index=dr)
        self.aggregate_direct_sum = pandas.DataFrame(ts, columns=['Value_sum'])
        self.aggregate_direct_sum.index.name = 'Datetime'

        self.aggregate_cli_mean = b"""Datetime,Value_mean
2011-01-01,2
2011-01-02,2
"""

        self.aggregate_cli_sum = b"""Datetime,Value_sum
2011-01-01,48
2011-01-02,48
"""

    def test_aggregate_direct_mean(self):
        out = tstoolbox.aggregate(statistic='mean',
                                  agg_interval='daily',
                                  input_ts='tests/data_flat.csv')
        assert_frame_equal(out, self.aggregate_direct_mean)

    def test_aggregate_direct_sum(self):
        out = tstoolbox.aggregate(statistic='sum',
                                  agg_interval='daily',
                                  input_ts='tests/data_flat.csv')
        assert_frame_equal(out, self.aggregate_direct_sum)

    def test_aggregate_cli_mean(self):
        args = ('tstoolbox aggregate '
                '--statistic="mean" '
                '--input_ts="tests/data_flat.csv"')
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]
        self.assertEqual(out, self.aggregate_cli_mean)

    def test_aggregate_cli_sum(self):
        args = ('tstoolbox aggregate '
                '--statistic="sum" '
                '--input_ts="tests/data_flat.csv"')
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]
        self.assertEqual(out, self.aggregate_cli_sum)
