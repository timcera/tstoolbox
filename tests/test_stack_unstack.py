# -*- coding: utf-8 -*-

from unittest import TestCase

import pandas as pd
from pandas.testing import assert_frame_equal

from tstoolbox import tstoolbox, tsutils


class TestRead(TestCase):
    def setUp(self):
        """Prepare in-memory files data_stacked.csv and data_unstacked.csv."""
        self.stacked = pd.read_csv(
            "tests/data_stacked.csv", index_col=0, parse_dates=True
        )
        self.stacked.index.name = "Datetime"

        self.stacked_1 = pd.read_csv(
            "tests/data_stacked_1.csv", index_col=0, parse_dates=True
        ).convert_dtypes()
        self.stacked_1.index.name = "Datetime"

        self.unstacked = pd.read_csv(
            "tests/data_unstacked.csv", index_col=0, parse_dates=True
        )
        self.unstacked.rename(columns=lambda x: x.strip("'\" "))
        self.unstacked.index.name = "Datetime"

    def test_stack(self):
        """Stack the data_unstacked.csv file.

        Compare against the in-memory version of the data_stacked.csv file.
        """
        out = tstoolbox.stack(input_ts="tests/data_unstacked.csv").convert_dtypes()
        assert_frame_equal(out, self.stacked_1)

    def test_unstack(self):
        """Unstack the data_stacked.csv file.

        Compare against the in-memory version of the data_unstacked.csv file.
        """
        out = tstoolbox.unstack("Columns", input_ts="tests/data_stacked.csv")
        assert_frame_equal(
            tsutils.memory_optimize(out), tsutils.memory_optimize(self.unstacked)
        )
