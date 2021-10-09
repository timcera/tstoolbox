# -*- coding: utf-8 -*-

from __future__ import print_function

from io import StringIO
from unittest import TestCase

import pandas as pd
from pandas.testing import assert_frame_equal

from tstoolbox import tstoolbox, tsutils

test_sinwave = r"""Datetime,Value
1999-12-31 23:00:00,0
2000-01-01 00:00:00,0.258819
2000-01-01 01:00:00,0.5
2000-01-01 02:00:00,0.707107
2000-01-01 03:00:00,0.866025
2000-01-01 04:00:00,0.965926
2000-01-01 05:00:00,1
2000-01-01 06:00:00,0.965926
2000-01-01 07:00:00,0.866025
2000-01-01 08:00:00,0.707107
2000-01-01 09:00:00,0.5
2000-01-01 10:00:00,0.258819
2000-01-01 11:00:00,1.22465e-16
2000-01-01 12:00:00,-0.258819
2000-01-01 13:00:00,-0.5
2000-01-01 14:00:00,-0.707107
2000-01-01 15:00:00,-0.866025
2000-01-01 16:00:00,-0.965926
2000-01-01 17:00:00,-1
2000-01-01 18:00:00,-0.965926
2000-01-01 19:00:00,-0.866025
2000-01-01 20:00:00,-0.707107
2000-01-01 21:00:00,-0.5
2000-01-01 22:00:00,-0.258819
"""


class TestDateOffset(TestCase):
    def setUp(self):
        self.ats = pd.read_csv(StringIO(test_sinwave), parse_dates=True, index_col=[0])
        self.ats = tsutils.memory_optimize(self.ats)

    def test_data_offset(self):
        out = tstoolbox.date_offset(-1, "H", input_ts="tests/data_sine.csv")
        assert_frame_equal(out, self.ats, check_dtype=False)
