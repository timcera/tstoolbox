# -*- coding: utf-8 -*-

from unittest import TestCase

import pandas
from pandas.testing import assert_frame_equal

from tstoolbox import tstoolbox, tsutils


class TestRead(TestCase):
    def setUp(self):
        dr = pandas.date_range("2000-01-01", periods=2, freq="D")

        ts = pandas.Series([4.5, 4.6], index=dr)

        self.read_direct = pandas.DataFrame(ts, columns=["Value"])
        self.read_direct.index.name = "Datetime"
        self.read_direct = tsutils.memory_optimize(self.read_direct)

        self.read_cli = b"""Datetime,Value
2000-01-01,4.5
2000-01-02,4.6
"""

        dr = pandas.date_range("2000-01-01", periods=5, freq="D")

        ts = pandas.Series([4.5, 4.6, 4.7, 4.8, 4.9], index=dr)

        self.read_direct_sparse = pandas.DataFrame(ts, columns=["Value"])
        self.read_direct_sparse.index.name = "Datetime"
        self.read_direct_sparse = tsutils.memory_optimize(self.read_direct_sparse)

        self.read_cli_sparse = b"""Datetime,Value
2000-01-01,4.5
2000-01-02,4.6
2000-01-03,4.7
2000-01-04,4.8
2000-01-05,4.9
"""

    def test_read_direct(self):
        """Test read API for single column - daily."""
        out = tstoolbox.read("tests/data_simple_extra_rows.csv", skiprows=2)
        assert_frame_equal(out, self.read_direct)

    def test_read_direct_sparse(self):
        """Test read API for single column - daily."""
        out = tstoolbox.read("tests/data_simple_extra_rows_sparse.csv", skiprows=[4, 6])
        assert_frame_equal(out, self.read_direct_sparse)
