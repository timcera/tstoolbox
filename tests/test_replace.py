# -*- coding: utf-8 -*-
from unittest import TestCase

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

import tstoolbox.tsutils as tsutils
from tstoolbox import tstoolbox

from . import capture


class TestReplace(TestCase):
    def setUp(self):
        dindex = pd.date_range("2011-01-01T00:00:00", periods=26, freq="H")
        self.ats = np.ones(26)
        self.ats = pd.DataFrame(
            self.ats, index=dindex, columns=["Value_with_missing::replace"]
        )
        self.ats.index.name = "Datetime"

        self.ats_cli = capture.capture(tsutils.printiso, self.ats)

        self.freplace_compare = self.ats.copy()
        self.freplace_compare["Value_with_missing::replace"][
            "2011-01-01T09:00:00":"2011-01-01T12:00:00"
        ] = 3
        self.freplace_compare["Value_with_missing::replace"]["2011-01-01T13:00:00"] = 9

        self.freplace_compare.columns = ["Value_with_missing"]
        self.freplace_compare_cli = capture.capture(
            tsutils.printiso, self.freplace_compare
        )

        self.breplace_compare = self.ats.copy()
        self.breplace_compare["Value_with_missing::replace"]["2011-01-01T09:00:00"] = 3
        self.breplace_compare["Value_with_missing::replace"][
            "2011-01-01T10:00:00":"2011-01-01T13:00:00"
        ] = 9

        self.breplace_compare.columns = ["Value_with_missing"]
        self.breplace_compare_cli = capture.capture(
            tsutils.printiso, self.breplace_compare
        )

        self.linear_compare = self.ats.copy()
        self.linear_compare["Value_with_missing::replace"]["2011-01-01T09:00:00"] = 3.0
        self.linear_compare["Value_with_missing::replace"]["2011-01-01T10:00:00"] = 4.5
        self.linear_compare["Value_with_missing::replace"]["2011-01-01T11:00:00"] = 6.0
        self.linear_compare["Value_with_missing::replace"]["2011-01-01T12:00:00"] = 7.5
        self.linear_compare["Value_with_missing::replace"]["2011-01-01T13:00:00"] = 9.0

        self.linear_compare.columns = ["Value_with_missing"]
        self.linear_compare_cli = capture.capture(tsutils.printiso, self.linear_compare)

        self.nearest_compare = self.ats.copy()
        self.nearest_compare["Value_with_missing::replace"][
            "2011-01-01T09:00:00":"2011-01-01T11:00:00"
        ] = 3.0
        self.nearest_compare["Value_with_missing::replace"][
            "2011-01-01T12:00:00":"2011-01-01T13:00:00"
        ] = 9.0

        self.nearest_compare.columns = ["Value_with_missing"]
        self.nearest_compare_cli = capture.capture(
            tsutils.printiso, self.nearest_compare
        )

        self.median_compare = self.ats.copy()
        self.median_compare["Value_with_missing::replace"]["2011-01-01T01:00:00"] = 2.0
        self.median_compare["Value_with_missing::replace"]["2011-01-01T09:00:00"] = 3.0
        self.median_compare["Value_with_missing::replace"]["2011-01-01T10:00:00"] = 2.0
        self.median_compare["Value_with_missing::replace"]["2011-01-01T11:00:00"] = 2.0
        self.median_compare["Value_with_missing::replace"]["2011-01-01T12:00:00"] = 2.0
        self.median_compare["Value_with_missing::replace"]["2011-01-01T13:00:00"] = 9.0
        self.median_compare["Value_with_missing::replace"]["2011-01-01T16:00:00"] = 2.0
        self.median_compare["Value_with_missing::replace"]["2011-01-01T22:00:00"] = 2.0
        self.median_compare["Value_with_missing::replace"]["2011-01-01T23:00:00"] = 2.0

        self.median_compare.columns = ["Value_with_missing"]
        self.median_compare_cli = capture.capture(tsutils.printiso, self.median_compare)

        self.max_compare = self.ats.copy()
        self.max_compare["Value_with_missing::replace"]["2011-01-01T01:00:00"] = 9.0
        self.max_compare["Value_with_missing::replace"]["2011-01-01T09:00:00"] = 3.0
        self.max_compare["Value_with_missing::replace"]["2011-01-01T10:00:00"] = 9.0
        self.max_compare["Value_with_missing::replace"]["2011-01-01T11:00:00"] = 9.0
        self.max_compare["Value_with_missing::replace"]["2011-01-01T12:00:00"] = 9.0
        self.max_compare["Value_with_missing::replace"]["2011-01-01T13:00:00"] = 9.0
        self.max_compare["Value_with_missing::replace"]["2011-01-01T16:00:00"] = 9.0
        self.max_compare["Value_with_missing::replace"]["2011-01-01T22:00:00"] = 9.0
        self.max_compare["Value_with_missing::replace"]["2011-01-01T23:00:00"] = 9.0

        self.max_compare.columns = ["Value_with_missing"]
        self.max_compare_cli = capture.capture(tsutils.printiso, self.max_compare)

        self.min_compare = self.ats.copy()
        self.min_compare["Value_with_missing::replace"]["2011-01-01T01:00:00"] = 2.0
        self.min_compare["Value_with_missing::replace"]["2011-01-01T09:00:00"] = 3.0
        self.min_compare["Value_with_missing::replace"]["2011-01-01T10:00:00"] = 2.0
        self.min_compare["Value_with_missing::replace"]["2011-01-01T11:00:00"] = 2.0
        self.min_compare["Value_with_missing::replace"]["2011-01-01T12:00:00"] = 2.0
        self.min_compare["Value_with_missing::replace"]["2011-01-01T13:00:00"] = 9.0
        self.min_compare["Value_with_missing::replace"]["2011-01-01T16:00:00"] = 2.0
        self.min_compare["Value_with_missing::replace"]["2011-01-01T22:00:00"] = 2.0
        self.min_compare["Value_with_missing::replace"]["2011-01-01T23:00:00"] = 2.0

        self.min_compare.columns = ["Value_with_missing"]
        self.min_compare_cli = capture.capture(tsutils.printiso, self.min_compare)

        self.con_compare = self.ats.copy()
        self.con_compare["Value_with_missing::replace"]["2011-01-01T01:00:00"] = 2.42
        self.con_compare["Value_with_missing::replace"]["2011-01-01T09:00:00"] = 3.0
        self.con_compare["Value_with_missing::replace"]["2011-01-01T10:00:00"] = 2.42
        self.con_compare["Value_with_missing::replace"]["2011-01-01T11:00:00"] = 2.42
        self.con_compare["Value_with_missing::replace"]["2011-01-01T12:00:00"] = 2.42
        self.con_compare["Value_with_missing::replace"]["2011-01-01T13:00:00"] = 9.0
        self.con_compare["Value_with_missing::replace"]["2011-01-01T16:00:00"] = 2.42
        self.con_compare["Value_with_missing::replace"]["2011-01-01T22:00:00"] = 2.42
        self.con_compare["Value_with_missing::replace"]["2011-01-01T23:00:00"] = 2.42

        self.con_compare.columns = ["Value_with_missing"]
        self.con_compare_cli = capture.capture(tsutils.printiso, self.con_compare)

    def test_replace_freplace_direct(self):
        """Test forward replace API."""
        out = tstoolbox.replace([3, 9], [1, 1], input_ts=self.freplace_compare)
        assert_frame_equal(out, self.ats, check_dtype=False)

    def test_replace_breplace(self):
        """Test backward replace API."""
        out = tstoolbox.replace([3, 9], [1, 1], input_ts=self.breplace_compare)
        assert_frame_equal(out, self.ats, check_dtype=False)

    def test_replace_linear(self):
        """Test linear interpolation replace API."""
        out = tstoolbox.replace(
            [3.0, 4.5, 6.0, 7.5, 9.0], [1, 1, 1, 1, 1], input_ts=self.linear_compare
        )
        assert_frame_equal(out, self.ats, check_dtype=False)

    def test_replace_nearest(self):
        """Test nearest replace API."""
        out = tstoolbox.replace([3.0, 9.0], [1, 1], input_ts=self.nearest_compare)
        assert_frame_equal(out, self.ats, check_dtype=False)

    def test_replace_median(self):
        """Test replace with median API."""
        out = tstoolbox.replace([2, 3, 9], [1, 1, 1], input_ts=self.median_compare)
        self.maxDiff = None
        assert_frame_equal(out, self.ats, check_dtype=False)

    def test_replace_max(self):
        """Test replace with max API."""
        out = tstoolbox.replace([3, 9], [1, 1], input_ts=self.max_compare)
        assert_frame_equal(out, self.ats, check_dtype=False)

    def test_replace_min(self):
        """Test replace with min API."""
        out = tstoolbox.replace([2, 3, 9], [1, 1, 1], input_ts=self.min_compare)
        assert_frame_equal(out, self.ats, check_dtype=False)

    def test_replace_con(self):
        """Test replace with con API."""
        out = tstoolbox.replace([2.42, 3, 9], [1, 1, 1], input_ts=self.con_compare)
        assert_frame_equal(out, self.ats, check_dtype=False)


#    def test_replace_freplace_cli(self):
#        """Test forward replace CLI."""
#        args = 'tstoolbox replace --input_ts=tests/data_missing.csv'
#        args = shlex.split(args)
#        out = subprocess.Popen(args,
#                               stdout=subprocess.PIPE,
#                               stdin=subprocess.PIPE).communicate(input=self.ats_cli)[0]
#        self.maxDiff = None
#        self.assertEqual(out, self.freplace_compare_cli)
#
#    def test_replace_breplace_cli(self):
#        """Test backward replace CLI."""
#        args = ('tstoolbox replace '
#                '--method="breplace" '
#                '--input_ts=tests/data_missing.csv')
#        args = shlex.split(args)
#        out = subprocess.Popen(args,
#                               stdout=subprocess.PIPE,
#                               stdin=subprocess.PIPE).communicate(input=self.ats_cli)[0]
#        self.maxDiff = None
#        self.assertEqual(out, self.breplace_compare_cli)
#
#    def test_replace_linear_cli(self):
#        """Test linear replace CLI."""
#        args = ('tstoolbox replace '
#                '--method="linear" '
#                '--input_ts=tests/data_missing.csv')
#        args = shlex.split(args)
#        out = subprocess.Popen(args,
#                               stdout=subprocess.PIPE,
#                               stdin=subprocess.PIPE).communicate(input=self.ats_cli)[0]
#        self.maxDiff = None
#        self.assertEqual(out, self.linear_compare_cli)
#
#    def test_replace_nearest_cli(self):
#        """Test nearest replace CLI."""
#        args = ('tstoolbox replace '
#                '--method="nearest" '
#                '--input_ts=tests/data_missing.csv')
#        args = shlex.split(args)
#        out = subprocess.Popen(args,
#                               stdout=subprocess.PIPE,
#                               stdin=subprocess.PIPE).communicate(input=self.ats_cli)[0]
#        self.maxDiff = None
#        self.assertEqual(out, self.nearest_compare_cli)
#
#    def test_replace_mean_cli(self):
#        """Test mean replace CLI."""
#        args = ('tstoolbox replace '
#                '--method="mean" '
#                '--input_ts=tests/data_missing.csv')
#        args = shlex.split(args)
#        out = subprocess.Popen(args,
#                               stdout=subprocess.PIPE,
#                               stdin=subprocess.PIPE).communicate(input=self.ats_cli)[0]
#        self.maxDiff = None
#        self.assertEqual(out, self.mean_compare_cli)
