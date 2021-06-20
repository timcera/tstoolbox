# -*- coding: utf-8 -*-

import shlex
import subprocess
from unittest import TestCase

import pandas
from pandas.testing import assert_frame_equal

from tstoolbox import tstoolbox, tsutils


class TestConvert(TestCase):
    def setUp(self):
        dr = pandas.date_range("2000-01-01", periods=2, freq="D")
        ts = pandas.Series([4.5, 4.6], index=dr)
        self.compare_direct_01 = pandas.DataFrame(ts, columns=["Value::convert"])
        self.compare_direct_01.index.name = "Datetime"
        self.compare_direct_01 = tsutils.memory_optimize(self.compare_direct_01)

        dr = pandas.date_range("2000-01-01", periods=2, freq="D")
        ts = pandas.Series([11.0, 11.2], index=dr)
        self.compare_direct_02 = pandas.DataFrame(ts, columns=["Value::convert"])
        self.compare_direct_02.index.name = "Datetime"
        self.compare_direct_02 = tsutils.memory_optimize(self.compare_direct_02)

        self.compare_cli_01 = b"""Datetime,Value::convert
2000-01-01,4.5
2000-01-02,4.6
"""
        self.compare_cli_02 = b"""Datetime,Value::convert
2000-01-01,11.0
2000-01-02,11.2
"""

    def test_convert_direct_01(self):
        """Test of convert API with default factor and offset."""
        out = tstoolbox.convert(input_ts="tests/data_simple.csv")
        assert_frame_equal(out, self.compare_direct_01)

    def test_convert_direct_02(self):
        """Test of convert API with set factor and offset."""
        out = tstoolbox.convert(input_ts="tests/data_simple.csv", factor=2, offset=2)
        assert_frame_equal(out, self.compare_direct_02)

    def test_convert_cli_01(self):
        """Test of CLI convert with default factor and offset."""
        args = 'tstoolbox convert --input_ts="tests/data_simple.csv"'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.compare_cli_01)

    def test_convert_cli_02(self):
        """Test of CLI convert set factor and offset."""
        args = (
            "tstoolbox convert "
            "--factor=2 "
            "--offset=2 "
            '--input_ts="tests/data_simple.csv"'
        )
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.compare_cli_02)
