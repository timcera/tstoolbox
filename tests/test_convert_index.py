#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import TestCase

import pandas as pd
from pandas.util.testing import assert_frame_equal
from tstoolbox import tstoolbox, tsutils

import pytest


class Testconvert_index(TestCase):
    def setUp(self):
        self.read_direct = pd.read_csv(
            "tests/data_gainesville_daily_precip_index.csv", index_col=0, header=0
        )
        self.read_direct = tsutils.memory_optimize(self.read_direct)

    def test_convert_index_from_UTC(self):
        out = tstoolbox.convert_index(
            "number",
            input_ts="tests/data_gainesville_daily_precip.csv",
            interval="D",
            epoch="1979-12-30",
        )
        out.index.name = "1979-12-30_date"
        assert_frame_equal(out, self.read_direct)


@pytest.mark.parametrize(
    "epoch,interval,expected",
    [
        ("1970-01-01", "D", 3653),
        ("unix", "D", 3653),
        ("unix", "s", 3653 * 24 * 60 * 60),
        ("1970-01-01", "H", 3653 * 24),
        ("julian", "D", 2444240.5),
        ("julian", "h", 2444240.5 * 24),
        ("reduced", "D", 2444240.5 - 2400000),
        ("modified", "D", 2444240.5 - 2400000.5),
        ("truncated", "D", pd.np.floor(2444240.5 - 2440000.5)),
        ("dublin", "D", 2444240.5 - 2415020),
        ("cnes", "D", 2444240.5 - 2433282.5),
        ("ccsds", "D", 2444240.5 - 2436204.5),
        ("lop", "D", 2444240.5 - 2448622.5),
        ("lilian", "D", pd.np.floor(2444240.5 - 2299159.5)),
        ("rata_die", "D", pd.np.floor(2444240.5 - 1721424.5)),
        ("mars_sol", "D", (2444240.5 - 2405522) / 1.02749),
        ("julian", None, 2444240.5),
    ],
)
def test_epoch_interval(epoch, interval, expected):
    out = tstoolbox.convert_index(
        "number",
        input_ts="tests/data_gainesville_daily_precip.csv",
        interval=interval,
        epoch=epoch,
    )
    assert out.index[1] == expected


def test_raises():
    with pytest.raises(ValueError):
        out = tstoolbox.convert_index(
            "zebra",
            input_ts="tests/data_gainesville_daily_precip.csv",
            interval="D",
            epoch="julian",
        )
