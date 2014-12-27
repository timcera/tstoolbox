#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_tstoolbox
----------------------------------

Tests for `tstoolbox` module.
"""

from pandas.util.testing import TestCase
from pandas.util.testing import assert_frame_equal
import subprocess
import shlex

import pandas

import tstoolbox.tsutils as tsutils
from tstoolbox import tstoolbox

from capture import capture

class TestRollingWindow(TestCase):
    def setUp(self):
        dr = pandas.date_range('2000-01-01', periods=2, freq='D')
        ts = pandas.TimeSeries([pandas.np.nan, 9.1], index=dr)
        self.compare_rolling_window_sum = pandas.DataFrame(ts,
                columns=['Value_sum'])
        self.compare_rolling_window_sum.index.name = 'Datetime'

        dr = pandas.date_range('2000-01-01', periods=2, freq='D')
        ts = pandas.TimeSeries([pandas.np.nan, 4.55], index=dr)
        self.compare_rolling_window_mean = pandas.DataFrame(
            ts, columns=['Value_mean'])
        self.compare_rolling_window_mean.index.name = 'Datetime'

        self.compare_rolling_window_sum_cli = capture(tsutils._printiso,
                self.compare_rolling_window_sum)

        self.compare_rolling_window_mean_cli = capture(tsutils._printiso,
                self.compare_rolling_window_mean)

    def test_rolling_window_sum(self):
        out = tstoolbox.rolling_window(statistic='sum',
                                       input_ts='tests/data_simple.csv')
        assert_frame_equal(out, self.compare_rolling_window_sum)

    def test_rolling_window_mean(self):
        out = tstoolbox.rolling_window(statistic='mean',
                                       input_ts='tests/data_simple.csv')
        assert_frame_equal(out, self.compare_rolling_window_mean)

    def test_rolling_window_sum_cli(self):
        args = ('tstoolbox rolling_window '
                '--statistic="sum" '
                '--input_ts="tests/data_simple.csv"')
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.compare_rolling_window_sum_cli)

    def test_rolling_window_mean_cli(self):
        args = ('tstoolbox rolling_window '
                '--statistic="mean" '
                '--input_ts="tests/data_simple.csv"')
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.compare_rolling_window_mean_cli)
