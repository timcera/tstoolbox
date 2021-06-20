# -*- coding: utf-8 -*-

from __future__ import print_function

import os
from unittest import TestCase

from pandas.testing import assert_frame_equal

from tstoolbox import tstoolbox

test_sinwave = """Datetime,0::,0::peak,0::valley
2000-01-01 00:00:00,0.0,,
2000-01-01 01:00:00,0.258819045103,,
2000-01-01 02:00:00,0.5,,
2000-01-01 03:00:00,0.707106781187,,
2000-01-01 04:00:00,0.866025403784,,
2000-01-01 05:00:00,0.965925826289,,
2000-01-01 06:00:00,1.0,1.0,
2000-01-01 07:00:00,0.965925826289,,
2000-01-01 08:00:00,0.866025403784,,
2000-01-01 09:00:00,0.707106781187,,
2000-01-01 10:00:00,0.5,,
2000-01-01 11:00:00,0.258819045103,,
2000-01-01 12:00:00,1.22464679915e-16,,
2000-01-01 13:00:00,-0.258819045103,,
2000-01-01 14:00:00,-0.5,,
2000-01-01 15:00:00,-0.707106781187,,
2000-01-01 16:00:00,-0.866025403784,,
2000-01-01 17:00:00,-0.965925826289,,
2000-01-01 18:00:00,-1.0,,-1.0
2000-01-01 19:00:00,-0.965925826289,,
2000-01-01 20:00:00,-0.866025403784,,
2000-01-01 21:00:00,-0.707106781187,,
2000-01-01 22:00:00,-0.5,,
2000-01-01 23:00:00,-0.258819045103,,
"""


class TestClean(TestCase):
    def setUp(self):
        self.ats = tstoolbox.read(os.path.join("tests", "data_sine.csv"))
        self.ats.index.name = "Datetime"
        self.ats.columns = ["Value"]

    def test_clean(self):
        out = tstoolbox.read("tests/data_dirty_index.csv", clean=True)
        self.maxDiff = None
        assert_frame_equal(out, self.ats)
