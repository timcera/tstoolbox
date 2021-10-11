# -*- coding: utf-8 -*-

import shlex
import subprocess
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import tstoolbox.tsutils as tsutils
from tstoolbox import tstoolbox

from . import capture


class TestFill(TestCase):
    def setUp(self):
        dindex = pd.date_range("2011-01-01T00:00:00", periods=26, freq="H")
        self.ats = np.ones(26) * 2
        self.ats = pd.DataFrame(
            self.ats,
            index=dindex,
            columns=["Value_with_missing::fill"],
            dtype="Float64",
        )
        self.ats.index.name = "Datetime"

        self.ats_cli = capture.capture(tsutils.printiso, self.ats)

        self.ffill_compare = self.ats.copy()
        self.ffill_compare["Value_with_missing::fill"][
            "2011-01-01T09:00:00":"2011-01-01T12:00:00"
        ] = 3
        self.ffill_compare["Value_with_missing::fill"]["2011-01-01T13:00:00"] = 9

        self.ffill_compare_cli = capture.capture(tsutils.printiso, self.ffill_compare)

        self.bfill_compare = self.ats.copy()
        self.bfill_compare["Value_with_missing::fill"]["2011-01-01T09:00:00"] = 3
        self.bfill_compare["Value_with_missing::fill"][
            "2011-01-01T10:00:00":"2011-01-01T13:00:00"
        ] = 9

        self.bfill_compare_cli = capture.capture(tsutils.printiso, self.bfill_compare)

        self.linear_compare = self.ats.copy()
        self.linear_compare["Value_with_missing::fill"]["2011-01-01T09:00:00"] = 3.0
        self.linear_compare["Value_with_missing::fill"]["2011-01-01T10:00:00"] = 4.5
        self.linear_compare["Value_with_missing::fill"]["2011-01-01T11:00:00"] = 6.0
        self.linear_compare["Value_with_missing::fill"]["2011-01-01T12:00:00"] = 7.5
        self.linear_compare["Value_with_missing::fill"]["2011-01-01T13:00:00"] = 9.0

        self.linear_compare_cli = capture.capture(tsutils.printiso, self.linear_compare)

        self.nearest_compare = self.ats.copy()
        self.nearest_compare["Value_with_missing::fill"][
            "2011-01-01T09:00:00":"2011-01-01T11:00:00"
        ] = 3.0
        self.nearest_compare["Value_with_missing::fill"][
            "2011-01-01T12:00:00":"2011-01-01T13:00:00"
        ] = 9.0
        self.nearest_compare = tsutils.memory_optimize(self.nearest_compare)

        self.nearest_compare_cli = capture.capture(
            tsutils.printiso, self.nearest_compare
        )

        self.mean_compare = self.ats.copy()
        self.mean_compare["Value_with_missing::fill"][
            "2011-01-01T01:00:00"
        ] = 2.4210526315789473
        self.mean_compare["Value_with_missing::fill"]["2011-01-01T09:00:00"] = 3.0
        self.mean_compare["Value_with_missing::fill"][
            "2011-01-01T10:00:00"
        ] = 2.4210526315789473
        self.mean_compare["Value_with_missing::fill"][
            "2011-01-01T11:00:00"
        ] = 2.4210526315789473
        self.mean_compare["Value_with_missing::fill"][
            "2011-01-01T12:00:00"
        ] = 2.4210526315789473
        self.mean_compare["Value_with_missing::fill"]["2011-01-01T13:00:00"] = 9.0
        self.mean_compare["Value_with_missing::fill"][
            "2011-01-01T16:00:00"
        ] = 2.4210526315789473
        self.mean_compare["Value_with_missing::fill"][
            "2011-01-01T22:00:00"
        ] = 2.4210526315789473
        self.mean_compare["Value_with_missing::fill"][
            "2011-01-01T23:00:00"
        ] = 2.4210526315789473

        self.mean_compare_cli = capture.capture(tsutils.printiso, self.mean_compare)

        self.median_compare = self.ats.copy()
        self.median_compare["Value_with_missing::fill"]["2011-01-01T01:00:00"] = 2.0
        self.median_compare["Value_with_missing::fill"]["2011-01-01T09:00:00"] = 3.0
        self.median_compare["Value_with_missing::fill"]["2011-01-01T10:00:00"] = 2.0
        self.median_compare["Value_with_missing::fill"]["2011-01-01T11:00:00"] = 2.0
        self.median_compare["Value_with_missing::fill"]["2011-01-01T12:00:00"] = 2.0
        self.median_compare["Value_with_missing::fill"]["2011-01-01T13:00:00"] = 9.0
        self.median_compare["Value_with_missing::fill"]["2011-01-01T16:00:00"] = 2.0
        self.median_compare["Value_with_missing::fill"]["2011-01-01T22:00:00"] = 2.0
        self.median_compare["Value_with_missing::fill"]["2011-01-01T23:00:00"] = 2.0
        self.median_compare = tsutils.memory_optimize(self.median_compare)

        self.median_compare_cli = capture.capture(tsutils.printiso, self.median_compare)

        self.max_compare = self.ats.copy()
        self.max_compare["Value_with_missing::fill"]["2011-01-01T01:00:00"] = 9.0
        self.max_compare["Value_with_missing::fill"]["2011-01-01T09:00:00"] = 3.0
        self.max_compare["Value_with_missing::fill"]["2011-01-01T10:00:00"] = 9.0
        self.max_compare["Value_with_missing::fill"]["2011-01-01T11:00:00"] = 9.0
        self.max_compare["Value_with_missing::fill"]["2011-01-01T12:00:00"] = 9.0
        self.max_compare["Value_with_missing::fill"]["2011-01-01T13:00:00"] = 9.0
        self.max_compare["Value_with_missing::fill"]["2011-01-01T16:00:00"] = 9.0
        self.max_compare["Value_with_missing::fill"]["2011-01-01T22:00:00"] = 9.0
        self.max_compare["Value_with_missing::fill"]["2011-01-01T23:00:00"] = 9.0
        self.max_compare = tsutils.memory_optimize(self.max_compare)

        self.max_compare_cli = capture.capture(tsutils.printiso, self.max_compare)

        self.min_compare = self.ats.copy()
        self.min_compare["Value_with_missing::fill"]["2011-01-01T01:00:00"] = 2.0
        self.min_compare["Value_with_missing::fill"]["2011-01-01T09:00:00"] = 3.0
        self.min_compare["Value_with_missing::fill"]["2011-01-01T10:00:00"] = 2.0
        self.min_compare["Value_with_missing::fill"]["2011-01-01T11:00:00"] = 2.0
        self.min_compare["Value_with_missing::fill"]["2011-01-01T12:00:00"] = 2.0
        self.min_compare["Value_with_missing::fill"]["2011-01-01T13:00:00"] = 9.0
        self.min_compare["Value_with_missing::fill"]["2011-01-01T16:00:00"] = 2.0
        self.min_compare["Value_with_missing::fill"]["2011-01-01T22:00:00"] = 2.0
        self.min_compare["Value_with_missing::fill"]["2011-01-01T23:00:00"] = 2.0
        self.min_compare = tsutils.memory_optimize(self.min_compare)

        self.min_compare_cli = capture.capture(tsutils.printiso, self.min_compare)

        self.con_compare = self.ats.copy()
        self.con_compare["Value_with_missing::fill"]["2011-01-01T01:00:00"] = 2.42
        self.con_compare["Value_with_missing::fill"]["2011-01-01T09:00:00"] = 3.0
        self.con_compare["Value_with_missing::fill"]["2011-01-01T10:00:00"] = 2.42
        self.con_compare["Value_with_missing::fill"]["2011-01-01T11:00:00"] = 2.42
        self.con_compare["Value_with_missing::fill"]["2011-01-01T12:00:00"] = 2.42
        self.con_compare["Value_with_missing::fill"]["2011-01-01T13:00:00"] = 9.0
        self.con_compare["Value_with_missing::fill"]["2011-01-01T16:00:00"] = 2.42
        self.con_compare["Value_with_missing::fill"]["2011-01-01T22:00:00"] = 2.42
        self.con_compare["Value_with_missing::fill"]["2011-01-01T23:00:00"] = 2.42

        self.con_compare_cli = capture.capture(tsutils.printiso, self.con_compare)

    def test_fill_ffill_direct(self):
        """Test forward fill API."""
        out = tstoolbox.fill(input_ts="tests/data_missing.csv")
        assert_frame_equal(out, self.ffill_compare)

    def test_fill_bfill(self):
        """Test backward fill API."""
        out = tstoolbox.fill(method="bfill", input_ts="tests/data_missing.csv")
        assert_frame_equal(out, self.bfill_compare)

    # def test_fill_linear(self):
    #     """Test linear interpolation fill API."""
    #     out = tstoolbox.fill(method="linear", input_ts="tests/data_missing.csv")
    #     assert_frame_equal(out, self.linear_compare)

    # def test_fill_nearest(self):
    #     """Test nearest fill API."""
    #     out = tstoolbox.fill(method="nearest", input_ts="tests/data_missing.csv")
    #     assert_frame_equal(out, self.nearest_compare)

    def test_fill_mean(self):
        """Test fill with mean API."""
        out = tstoolbox.fill(method="mean", input_ts="tests/data_missing.csv")
        assert_frame_equal(out, self.mean_compare)

    def test_fill_median(self):
        """Test fill with median API."""
        out = tstoolbox.fill(method="median", input_ts="tests/data_missing.csv")
        assert_frame_equal(out, self.median_compare)

    def test_fill_max(self):
        """Test fill with max API."""
        out = tstoolbox.fill(method="max", input_ts="tests/data_missing.csv")
        assert_frame_equal(out, self.max_compare)

    def test_fill_min(self):
        """Test fill with min API."""
        out = tstoolbox.fill(method="min", input_ts="tests/data_missing.csv")
        assert_frame_equal(out, self.min_compare)

    def test_fill_con(self):
        """Test fill with con API."""
        out = tstoolbox.fill(method=2.42, input_ts="tests/data_missing.csv")
        assert_frame_equal(out, self.con_compare)

    @staticmethod
    def test_fill_from():
        """Test fill with from API."""
        out = tstoolbox.fill(
            method="from",
            input_ts="tests/data_fill_from.csv",
            from_columns=[2, 3],
            to_columns=1,
        ).astype("Float64")
        compare = tstoolbox.read("tests/data_fill_from.csv").astype("Float64")
        compare.columns = ["Value::fill", "Value1::fill", "Value2::fill"]
        compare.loc["2000-01-02", "Value::fill"] = 2.5
        compare.loc["2000-01-04", "Value::fill"] = 23.1
        assert_frame_equal(out, compare)

    @staticmethod
    def test_fill_value():
        """Test fill with value API."""
        with pytest.raises(ValueError) as e_info:
            _ = tstoolbox.fill(method="a", input_ts="tests/data_missing.csv")
        assert r"could not convert " in str(e_info.value)

    def test_fill_ffill_cli(self):
        """Test forward fill CLI."""
        args = "tstoolbox fill --input_ts=tests/data_missing.csv"
        args = shlex.split(args)
        out = subprocess.Popen(
            args, stdout=subprocess.PIPE, stdin=subprocess.PIPE
        ).communicate(input=self.ats_cli)[0]
        self.maxDiff = None
        self.assertEqual(out, self.ffill_compare_cli)

    def test_fill_bfill_cli(self):
        """Test backward fill CLI."""
        args = "tstoolbox fill " '--method="bfill" ' "--input_ts=tests/data_missing.csv"
        args = shlex.split(args)
        out = subprocess.Popen(
            args, stdout=subprocess.PIPE, stdin=subprocess.PIPE
        ).communicate(input=self.ats_cli)[0]
        self.maxDiff = None
        self.assertEqual(out, self.bfill_compare_cli)

    # def test_fill_linear_cli(self):
    #     """Test linear fill CLI."""
    #     args = (
    #         "tstoolbox fill "
    #         '--method="linear" '
    #         "--input_ts=tests/data_missing.csv"
    #     )
    #     args = shlex.split(args)
    #     out = subprocess.Popen(
    #         args, stdout=subprocess.PIPE, stdin=subprocess.PIPE
    #     ).communicate(input=self.ats_cli)[0]
    #     self.maxDiff = None
    #     self.assertEqual(out, self.linear_compare_cli)

    # def test_fill_nearest_cli(self):
    #     """Test nearest fill CLI."""
    #     args = (
    #         "tstoolbox fill "
    #         '--method="nearest" '
    #         "--input_ts=tests/data_missing.csv"
    #     )
    #     args = shlex.split(args)
    #     out = subprocess.Popen(
    #         args, stdout=subprocess.PIPE, stdin=subprocess.PIPE
    #     ).communicate(input=self.ats_cli)[0]
    #     self.maxDiff = None
    #     self.assertEqual(out, self.nearest_compare_cli)

    def test_fill_mean_cli(self):
        """Test mean fill CLI."""
        args = "tstoolbox fill " '--method="mean" ' "--input_ts=tests/data_missing.csv"
        args = shlex.split(args)
        out = subprocess.Popen(
            args, stdout=subprocess.PIPE, stdin=subprocess.PIPE
        ).communicate(input=self.ats_cli)[0]
        self.maxDiff = None
        self.assertEqual(out, self.mean_compare_cli)
