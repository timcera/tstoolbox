#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pick
----------------------------------

Tests for `tstoolbox` module.
"""

from pandas.util.testing import TestCase
from pandas.util.testing import assert_frame_equal
import shlex
import subprocess
import pandas

from tstoolbox import tstoolbox
import tstoolbox.tsutils as tsutils

from capture import capture

class TestPick(TestCase):
    def setUp(self):
        dr = pandas.date_range('2000-01-01', periods=6, freq='D')

        ts1 = pandas.TimeSeries([4.5, 4.6, 4.7, 4.6, 4.5, 4.4],
                index=dr)
        ts2 = pandas.TimeSeries([45.6, 90.5, 34.2, 23.1, 7.2, 4.3],
                index=dr)
        self.pick_multiple_direct = pandas.DataFrame(ts2,
                columns=['Value1'])
        self.pick_multiple_direct = self.pick_multiple_direct.join(
            pandas.DataFrame(ts1, columns=['Value']))

        self.pick_cli = capture(tsutils._printiso, self.pick_multiple_direct)

    def test_pick(self):
        out = tstoolbox.pick('2,1', input_ts='tests/data_multiple_cols.csv')
        assert_frame_equal(out, self.pick_multiple_direct)

    def test_pick_cli(self):
        args = 'tstoolbox pick 2,1 --input_ts="tests/data_multiple_cols.csv"'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.pick_cli)
