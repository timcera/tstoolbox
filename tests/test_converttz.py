# -*- coding: utf-8 -*-

from unittest import TestCase

import pandas as pd
from pandas.testing import assert_frame_equal

from tstoolbox import tstoolbox, tsutils


class TestConvertTZ(TestCase):
    def setUp(self):
        try:
            self.read_direct = (
                pd.read_csv("tests/data_sunspot_EST.csv", index_col=0, parse_dates=[0])
                .tz_localize("UTC")
                .tz_convert("EST")
            )
        except TypeError:
            self.read_direct = pd.read_csv(
                "tests/data_sunspot_EST.csv", index_col=0, parse_dates=[0]
            ).tz_convert("EST")
        self.read_direct = tsutils.memory_optimize(self.read_direct)

    def test_converttz_from_UTC(self):
        out = tstoolbox.converttz("UTC", "EST", input_ts="tests/data_sunspot.csv")
        out.index.name = "Datetime"
        assert_frame_equal(out, self.read_direct)
