# -*- coding: utf-8 -*-

import shlex
import subprocess
from unittest import TestCase

import pandas
from pandas.testing import assert_frame_equal

from tstoolbox import tstoolbox, tsutils


class Testround_index(TestCase):
    def setUp(self):
        dr = pandas.date_range("2000-01-01", periods=2, freq="D")

        ts = pandas.Series([4.5, 4.6], index=dr)

        self.round_index_direct = pandas.DataFrame(ts, columns=["Value"])
        self.round_index_direct.index.name = "Datetime"
        self.round_index_direct = tsutils.memory_optimize(self.round_index_direct)

        self.round_index_multiple_direct = pandas.DataFrame(ts, columns=["Value"])
        self.round_index_multiple_direct = pandas.concat(
            [self.round_index_multiple_direct, pandas.Series(ts, name="Value_r")],
            axis="columns",
        )
        self.round_index_multiple_direct.index.name = "Datetime"
        self.round_index_multiple_direct = tsutils.memory_optimize(
            self.round_index_multiple_direct
        )

        self.round_index_cli = b"""Datetime,Value
2000-01-01,4.5
2000-01-02,4.6
"""

        self.round_index_multiple_cli = b"""Datetime,Value,Value_r
2000-01-01,4.5,4.5
2000-01-02,4.6,4.6
"""

        self.round_index_tsstep_2_daily_cli = b"""Datetime,Value,Value1
2000-01-01,4.5,45.6
2000-01-03,4.7,34.2
2000-01-05,4.5,7.2
"""
        self.round_index_tsstep_2_daily = pandas.DataFrame(
            [[4.5, 45.6], [4.7, 34.2], [4.5, 7.2]],
            columns=["Value", "Value1"],
            index=pandas.DatetimeIndex(["2000-01-01", "2000-01-03", "2000-01-05"]),
        )
        self.round_index_tsstep_2_daily = tsutils.memory_optimize(
            self.round_index_tsstep_2_daily
        )

        self.round_index_tsstep_2_daily.index.name = "Datetime"

        self.round_index_blanks = b"""Datetime,Value_mean,Unnamed: 2_mean,Unnamed: 3_mean,Unnamed: 4_mean,Unnamed: 5_mean,Unnamed: 6_mean,Unnamed: 7_mean,Unnamed: 8_mean,Unnamed: 9_mean
2000-01-01,2.46667,,,,,,,,
2000-01-02,3.4,,,,,,,,
"""

    def test_round_index_direct(self):
        """Test round_index API for single column - daily."""
        out = tstoolbox.read("tests/data_simple.csv", round_index="D")
        assert_frame_equal(out, self.round_index_direct)

    def test_round_index_mulitple_direct(self):
        """Test round_index API for multiple columns - daily."""
        out = tstoolbox.read(
            "tests/data_simple.csv tests/data_simple.csv",
            append="columns",
            round_index="D",
        )
        assert_frame_equal(out, self.round_index_multiple_direct)

    def test_round_index_bi_monthly(self):
        """Test round_index API for bi monthly time series."""
        out = tstoolbox.read("tests/data_bi_daily.csv", round_index="D")
        assert_frame_equal(out, self.round_index_tsstep_2_daily)

    def test_round_index_cli(self):
        """Test round_index CLI for single column - daily."""
        args = 'tstoolbox read --round_index="D" tests/data_simple.csv'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.round_index_cli)

    def test_round_index_multiple_cli(self):
        """Test round_index CLI for multiple columns - daily."""
        args = 'tstoolbox read --round_index="D" tests/data_simple.csv tests/data_simple.csv'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.round_index_multiple_cli)

    def test_round_index_bi_monthly_cli(self):
        """Test round_index CLI for bi monthly time series."""
        args = 'tstoolbox read --round_index="D" tests/data_bi_daily.csv'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.round_index_tsstep_2_daily_cli)
