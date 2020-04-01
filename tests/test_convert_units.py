#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import TestCase
import pytest

from pandas.testing import assert_frame_equal
from tstoolbox import tstoolbox


class TestConvertUnits(TestCase):
    def test_convert_units(self):
        a = tstoolbox.read("tests/data_gainesville_daily_precip.csv", target_units="in")
        b = tstoolbox.equation(
            "x1/25.4", input_ts="tests/data_gainesville_daily_precip.csv"
        ).astype("float64")
        b.columns = ["ADaymet-prcp:in"]
        assert_frame_equal(a, b)

        a = tstoolbox.read("tests/data_gainesville_daily_precip.csv", target_units="km")
        b = tstoolbox.equation(
            "x1/(1000*1000)", input_ts="tests/data_gainesville_daily_precip.csv"
        ).astype("float64")
        b.columns = ["ADaymet-prcp:km"]
        assert_frame_equal(a, b)

        with pytest.raises(ValueError) as e_info:
            _ = tstoolbox.read(
                "tests/data_gainesville_daily_precip.csv", source_units="ft3/s"
            )
        assert (
            r"   You specified 'source_units' as ft3/s, but column units are mm."
            in str(e_info.value)
        )

        with pytest.raises(ValueError) as e_info:
            _ = tstoolbox.read(
                "tests/data_gainesville_daily_precip.csv", target_units="ft3/s"
            )
