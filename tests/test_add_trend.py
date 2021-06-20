# -*- coding: utf-8 -*-

import shlex
import subprocess
from unittest import TestCase

from pandas.testing import assert_frame_equal

from tstoolbox import tstoolbox, tsutils


class TestAddTrend(TestCase):
    def setUp(self):
        self.add_trend_cli = b"""Datetime,Value::trend
2011-01-01 00:00:00,1
2011-01-01 01:00:00,1.04255
2011-01-01 02:00:00,1.08511
2011-01-01 03:00:00,1.12766
2011-01-01 04:00:00,1.17021
2011-01-01 05:00:00,1.21277
2011-01-01 06:00:00,1.25532
2011-01-01 07:00:00,1.29787
2011-01-01 08:00:00,1.34043
2011-01-01 09:00:00,1.38298
2011-01-01 10:00:00,1.42553
2011-01-01 11:00:00,1.46809
2011-01-01 12:00:00,1.51064
2011-01-01 13:00:00,1.55319
2011-01-01 14:00:00,1.59574
2011-01-01 15:00:00,1.6383
2011-01-01 16:00:00,1.68085
2011-01-01 17:00:00,1.7234
2011-01-01 18:00:00,1.76596
2011-01-01 19:00:00,1.80851
2011-01-01 20:00:00,1.85106
2011-01-01 21:00:00,1.89362
2011-01-01 22:00:00,1.93617
2011-01-01 23:00:00,1.97872
2011-01-02 00:00:00,2.02128
2011-01-02 01:00:00,2.06383
2011-01-02 02:00:00,2.10638
2011-01-02 03:00:00,2.14894
2011-01-02 04:00:00,2.19149
2011-01-02 05:00:00,2.23404
2011-01-02 06:00:00,2.2766
2011-01-02 07:00:00,2.31915
2011-01-02 08:00:00,2.3617
2011-01-02 09:00:00,2.40426
2011-01-02 10:00:00,2.44681
2011-01-02 11:00:00,2.48936
2011-01-02 12:00:00,2.53191
2011-01-02 13:00:00,2.57447
2011-01-02 14:00:00,2.61702
2011-01-02 15:00:00,2.65957
2011-01-02 16:00:00,2.70213
2011-01-02 17:00:00,2.74468
2011-01-02 18:00:00,2.78723
2011-01-02 19:00:00,2.82979
2011-01-02 20:00:00,2.87234
2011-01-02 21:00:00,2.91489
2011-01-02 22:00:00,2.95745
2011-01-02 23:00:00,3
"""
        self.add_trend_direct = tstoolbox.date_slice(input_ts=self.add_trend_cli)
        self.add_trend_direct.index.name = "Datetime"
        self.add_trend_direct = tsutils.memory_optimize(self.add_trend_direct)

    def test_add_trend_direct(self):
        """Add trend using API."""
        out = tstoolbox.add_trend(-1.0, 1.0, input_ts="tests/data_flat.csv")
        assert_frame_equal(out, self.add_trend_direct)

    def test_add_trend_cli(self):
        """Add trend using the CLI."""
        args = 'tstoolbox add_trend -1 1 --input_ts="tests/data_flat.csv"'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]
        out = tstoolbox.date_slice(input_ts=out)
        assert_frame_equal(out, self.add_trend_direct)
