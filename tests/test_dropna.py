from unittest import TestCase

import pandas as pd
from pandas.util.testing import assert_frame_equal
from tstoolbox import tstoolbox


class TestRead(TestCase):
    def setUp(self):
        self.read_direct = pd.read_csv(
            "tests/data_missing.csv",
            index_col=[0],
            parse_dates=True,
            skipinitialspace=True,
        ).astype("float64")
        self.read_direct.index.name = "Datetime"

    def test_read_direct_dropna(self):
        """Test read dropna for single column - daily."""
        out = tstoolbox.read("tests/data_missing.csv", dropna="all")
        assert_frame_equal(out, self.read_direct)
