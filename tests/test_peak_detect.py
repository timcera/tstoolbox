# -*- coding: utf-8 -*-

import shlex
import subprocess
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tstoolbox import tstoolbox, tsutils

output_peak_detection = tsutils.read_iso_ts(
    b"""Datetime,0,0::peak,0::valley
2000-01-01 00:00:00,0,,
2000-01-01 01:00:00,0.258819,,
2000-01-01 02:00:00,0.5,,
2000-01-01 03:00:00,0.707107,,
2000-01-01 04:00:00,0.866025,,
2000-01-01 05:00:00,0.965926,,
2000-01-01 06:00:00,1,1,
2000-01-01 07:00:00,0.965926,,
2000-01-01 08:00:00,0.866025,,
2000-01-01 09:00:00,0.707107,,
2000-01-01 10:00:00,0.5,,
2000-01-01 11:00:00,0.258819,,
2000-01-01 12:00:00,1.22465e-16,,
2000-01-01 13:00:00,-0.258819,,
2000-01-01 14:00:00,-0.5,,
2000-01-01 15:00:00,-0.707107,,
2000-01-01 16:00:00,-0.866025,,
2000-01-01 17:00:00,-0.965926,,
2000-01-01 18:00:00,-1,,-1
2000-01-01 19:00:00,-0.965926,,
2000-01-01 20:00:00,-0.866025,,
2000-01-01 21:00:00,-0.707107,,
2000-01-01 22:00:00,-0.5,,
2000-01-01 23:00:00,-0.258819,,
"""
)

input_peak_detection = b"""Datetime,0
2000-01-01 00:00:00,0.0
2000-01-01 01:00:00,0.258819
2000-01-01 02:00:00,0.5
2000-01-01 03:00:00,0.707107
2000-01-01 04:00:00,0.866025
2000-01-01 05:00:00,0.965926
2000-01-01 06:00:00,1.0
2000-01-01 07:00:00,0.965926
2000-01-01 08:00:00,0.866025
2000-01-01 09:00:00,0.707107
2000-01-01 10:00:00,0.5
2000-01-01 11:00:00,0.258819
2000-01-01 12:00:00,1.22465e-16
2000-01-01 13:00:00,-0.258819
2000-01-01 14:00:00,-0.5
2000-01-01 15:00:00,-0.707107
2000-01-01 16:00:00,-0.866025
2000-01-01 17:00:00,-0.965926
2000-01-01 18:00:00,-1.0
2000-01-01 19:00:00,-0.965926
2000-01-01 20:00:00,-0.866025
2000-01-01 21:00:00,-0.707107
2000-01-01 22:00:00,-0.5
2000-01-01 23:00:00,-0.258819
"""


class TestPeakDetect(TestCase):
    def setUp(self):
        dindex = pd.date_range("2000-01-01T00:00:00", periods=24, freq="H")
        self.ats = np.arange(0, 360, 15)
        self.ats = np.sin(2 * np.pi * self.ats / 360)
        self.ats = pd.DataFrame(self.ats, index=dindex)
        self.ats = tsutils.memory_optimize(self.ats).astype("Float64")

        self.compare = self.ats.copy()
        self.compare = self.compare.join(
            pd.Series(
                np.zeros(len(self.ats)).astype("f"),
                index=self.ats.index,
                name="0::peak",
            )
        )
        self.compare = self.compare.join(
            pd.Series(
                np.zeros(len(self.ats)).astype("f"),
                index=self.ats.index,
                name="0::valley",
            )
        )
        self.compare.index.name = "Datetime"
        self.compare["0::peak"] = np.nan
        self.compare.loc[self.compare[0] == 1, "0::peak"] = 1
        self.compare["0::valley"] = np.nan
        self.compare.loc[self.compare[0] == -1, "0::valley"] = -1
        self.compare = tsutils.memory_optimize(self.compare).astype("Float64")

    def test_peak_rel_direct(self):
        """Test peak detection API using the default method."""
        out = tstoolbox.peak_detection(
            input_ts=self.ats, print_input=True, extrema="both"
        )
        cols = list(out.columns)
        cols[0] = 0
        out.columns = cols
        assert_frame_equal(out, self.compare)

    def test_peak_minmax_direct(self):
        """Test peak detection API using the minmax method."""
        out = tstoolbox.peak_detection(
            method="minmax",
            window=3,
            input_ts=self.ats,
            print_input=True,
            extrema="both",
        )
        cols = list(out.columns)
        cols[0] = 0
        out.columns = cols
        assert_frame_equal(out, self.compare)

    def test_peak_zero_crossing_direct(self):
        """Test peak detection API using the zero_crossing method."""
        out = tstoolbox.peak_detection(
            method="zero_crossing",
            window=3,
            input_ts=self.ats,
            print_input=True,
            extrema="both",
        )
        cols = list(out.columns)
        cols[0] = 0
        out.columns = cols
        assert_frame_equal(out, self.compare)

    #    def test_peak_parabola_direct(self):
    #        out = tstoolbox.peak_detection(method='parabola',
    #                                       input_ts=self.ats,
    #                                       print_input=True,
    #                                       extrema='both')
    #        self.maxDiff = None
    #        assert_frame_equal(out, self.compare)

    def test_peak_sine_direct(self):
        """Test peak detection API using the 'sine' method."""
        out = tstoolbox.peak_detection(
            method="sine", points=9, input_ts=self.ats, print_input=True, extrema="both"
        )
        cols = list(out.columns)
        cols[0] = 0
        out.columns = cols
        assert_frame_equal(out, self.compare)

    # CLI...
    @staticmethod
    def test_peak_rel_cli():
        """Test peak detection CLI using the default method."""
        args = 'tstoolbox peak_detection --extrema="both" --print_input=True'
        args = shlex.split(args)
        out = subprocess.Popen(
            args, stdout=subprocess.PIPE, stdin=subprocess.PIPE
        ).communicate(input=input_peak_detection)[0]
        out = tsutils.read_iso_ts(out)
        assert_frame_equal(out, output_peak_detection)

    @staticmethod
    def test_peak_minmax_cli():
        """Test peak detection CLI using the minmax method."""
        args = (
            "tstoolbox peak_detection "
            "--window=3 "
            '--method="minmax" '
            '--extrema="both" '
            "--print_input=True"
        )
        args = shlex.split(args)
        out = subprocess.Popen(
            args, stdout=subprocess.PIPE, stdin=subprocess.PIPE
        ).communicate(input=input_peak_detection)[0]
        out = tsutils.read_iso_ts(out)
        # input_peak_detection.to_csv("input.csv")
        output_peak_detection.to_csv("output.csv")
        out.to_csv("out.csv")
        assert_frame_equal(out, output_peak_detection)

    @staticmethod
    def test_peak_zero_crossing_cli():
        """Test peak detection CLI using the zero_crossing method."""
        args = (
            "tstoolbox peak_detection "
            '--method="zero_crossing" '
            '--extrema="both" '
            "--window=3 "
            "--print_input=True"
        )
        args = shlex.split(args)
        out = subprocess.Popen(
            args, stdout=subprocess.PIPE, stdin=subprocess.PIPE
        ).communicate(input=input_peak_detection)[0]
        out = tsutils.read_iso_ts(out)
        assert_frame_equal(out, output_peak_detection)

    # def test_peak_parabola_cli(self):
    #     args = ('tstoolbox peak_detection '
    #             '--method="parabola" --extrema="both" --print_input=True')
    #     args = shlex.split(args)
    #     out = subprocess.Popen(args,
    #         stdout=subprocess.PIPE,
    #         stdin=subprocess.PIPE).communicate(input=input_peak_detection)[0]
    #     self.maxDiff = None
    #     self.assertEqual(out, output_peak_detection)

    @staticmethod
    def test_peak_sine_cli():
        """Test peak detection CLI using the 'sine' method."""
        args = (
            "tstoolbox peak_detection "
            '--method="sine" '
            '--extrema="both" '
            "--points=9 "
            "--print_input=True"
        )
        args = shlex.split(args)
        out = subprocess.Popen(
            args, stdout=subprocess.PIPE, stdin=subprocess.PIPE
        ).communicate(input=input_peak_detection)[0]
        out = tsutils.read_iso_ts(out)
        assert_frame_equal(out, output_peak_detection)

    def test_peak_type_error(self):
        with pytest.raises(ValueError) as e_info:
            _ = tstoolbox.peak_detection(
                method="sine",
                points=9,
                input_ts=self.ats,
                print_input=True,
                extrema="booth",
            )
        assert "Given value <'booth'> fails constraints:" in str(e_info.value)

        with pytest.raises(ValueError) as e_info:
            _ = tstoolbox.peak_detection(
                method="sin",
                points=9,
                input_ts=self.ats,
                print_input=True,
                extrema="both",
            )
        assert "Given value <'sin'> fails constraints:" in str(e_info.value)
