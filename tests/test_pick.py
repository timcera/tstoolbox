# -*- coding: utf-8 -*-

import shlex
import subprocess
from unittest import TestCase

import pandas
from pandas.testing import assert_frame_equal

from tstoolbox import tstoolbox, tsutils

from . import capture


class TestPick(TestCase):
    def setUp(self):
        dr = pandas.date_range("2000-01-01", periods=6, freq="D")

        ts1 = pandas.Series([4.5, 4.6, 4.7, 4.6, 4.5, 4.4], index=dr)
        ts2 = pandas.Series([45.6, 90.5, 34.2, 23.1, 7.2, 4.3], index=dr)
        self.pick_multiple_direct = pandas.DataFrame(ts2, columns=["Value1"])
        self.pick_multiple_direct = self.pick_multiple_direct.join(
            pandas.DataFrame(ts1, columns=["Value"])
        )
        self.pick_multiple_direct = tsutils.memory_optimize(self.pick_multiple_direct)

        self.pick_cli = capture.capture(tsutils.printiso, self.pick_multiple_direct)

    def test_pick(self):
        """Test the pick API by picking the 2nd then the 1st column.

        Reversing the order of the columns.
        """
        out = tstoolbox.pick("2,1", input_ts="tests/data_multiple_cols.csv")
        assert_frame_equal(out, self.pick_multiple_direct)

    def test_pick_cli(self):
        """Test the pick API by picking the 2nd then the 1st column.

        Effectively reversing the order of the columns.
        """
        args = 'tstoolbox pick 2,1 --input_ts="tests/data_multiple_cols.csv"'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.pick_cli)
