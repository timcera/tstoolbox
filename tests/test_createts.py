# -*- coding: utf-8 -*-

from __future__ import print_function

from unittest import TestCase

import pandas as pd
import pytest
from pandas.testing import assert_index_equal

from tstoolbox import tstoolbox, tsutils


class TestRead(TestCase):
    def setUp(self):
        """Prepare in-memory versions of the files ./data_flat.csv."""
        self.data = pd.read_csv(
            "tests/data_flat.csv", sep=",", parse_dates=True, index_col=0
        )
        self.data.index.name = "Datetime"

    def test_createts_from_input(self):
        """Create a ts of data_flat.csv."""
        out = tstoolbox.createts(input_ts=self.data).index
        out1 = tstoolbox.read("tests/data_flat.csv").index
        assert_index_equal(out, out1)

    def test_createts_from_dates(self):
        """Create a ts file from start/end dates and freq."""
        sdate = self.data.index[0]
        edate = self.data.index[-1]
        freq = tsutils.asbestfreq(self.data)
        freq = freq.index.freqstr
        out = tstoolbox.createts(start_date=sdate, end_date=edate, freq=freq).index
        out.name = "Datetime"
        assert_index_equal(out, self.data.index)

    @staticmethod
    def test_exception():
        with pytest.raises(ValueError) as e_info:
            _ = tstoolbox.createts()
        assert r"If input_ts is None, then start_date, end_date" in str(e_info.value)
