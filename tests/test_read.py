#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_read
----------------------------------

Tests for `tstoolbox` module.
"""

from pandas.util.testing import TestCase
from pandas.util.testing import assert_frame_equal
import shlex
import subprocess

import pandas

from tstoolbox import tstoolbox

from capture import capture

class TestRead(TestCase):
    def setUp(self):
        dr = pandas.date_range('2000-01-01', periods=2, freq='D')

        ts = pandas.TimeSeries([4.5, 4.6], index=dr)

        self.read_direct = pandas.DataFrame(ts, columns=['Value'])
        self.read_direct.index.name = 'Datetime'

        self.read_multiple_direct = pandas.DataFrame(ts, columns=['Value'])
        self.read_multiple_direct = self.read_multiple_direct.join(
            pandas.Series(ts, name='Value_data_simple'))
        self.read_multiple_direct.index.name = 'Datetime'

        self.read_cli = b"""Datetime,Value
2000-01-01,4.5
2000-01-02,4.6
"""

        self.read_multiple_cli = b"""Datetime,Value,Value_data_simple
2000-01-01,4.5,4.5
2000-01-02,4.6,4.6
"""

        self.read_tsstep_2_daily_cli = b"""Datetime,Value,Value1
2000-01-01,4.5,45.6
2000-01-03,4.7,34.2
2000-01-05,4.5,7.2
"""
        self.read_tsstep_2_daily = pandas.DataFrame([[4.5, 45.6], [4.7, 34.2],
            [4.5, 7.2]], columns=['Value', 'Value1'], index=['2000-01-01',
            '2000-01-03', '2000-01-05'])
        self.read_tsstep_2_daily.index.name = 'Datetime'

    def test_read_direct(self):
        out = tstoolbox.read('tests/data_simple.csv')
        assert_frame_equal(out, self.read_direct)

    def test_read_mulitple_direct(self):
        out = tstoolbox.read('tests/data_simple.csv,tests/data_simple.csv')
        assert_frame_equal(out, self.read_multiple_direct)

    def test_read_bi_monthly(self):
        out = tstoolbox.read('tests/data_bi_daily.csv')
        assert_frame_equal(out, self.read_tsstep_2_daily)

    def test_read_cli(self):
        args = 'tstoolbox read tests/data_simple.csv'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.read_cli)

    def test_read_mulitple_cli(self):
        args = 'tstoolbox read tests/data_simple.csv,tests/data_simple.csv'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.read_multiple_cli)

    def test_read_bi_monthly_cli(self):
        args = 'tstoolbox read tests/data_bi_daily.csv'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.read_tsstep_2_daily_cli)
