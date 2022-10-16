import shlex
import subprocess
from io import BytesIO
from unittest import TestCase

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from toolbox_utils import tsutils

from tstoolbox import tstoolbox


class TestAggregate(TestCase):
    def setUp(self):
        dr = pd.date_range("2011-01-01", periods=2, freq="D")

        ts = pd.Series([2, 2], index=dr)
        self.aggregate_direct_mean = pd.DataFrame(ts, columns=["Value::mean"])
        self.aggregate_direct_mean.index.name = "Datetime"
        self.aggregate_direct_mean = tsutils.memory_optimize(
            self.aggregate_direct_mean
        ).astype("Int64")

        ts = pd.Series([48.0, 48.0], index=dr)
        self.aggregate_direct_sum = pd.DataFrame(ts, columns=["Value::sum"])
        self.aggregate_direct_sum.index.name = "Datetime"
        self.aggregate_direct_sum = tsutils.memory_optimize(
            self.aggregate_direct_sum
        ).astype("Float64")

        self.aggregate_cli_mean = b"""Datetime,Value::mean
2011-01-01,2.0
2011-01-02,2.0
"""

        self.aggregate_cli_sum = b"""Datetime,Value::sum
2011-01-01,48
2011-01-02,48
"""

    def test_aggregate_direct_mean(self):
        """Test daily mean aggregation."""
        out = tstoolbox.aggregate(
            statistic="mean", groupby="daily", input_ts="tests/data_flat.csv"
        ).astype("Float64")
        assert_frame_equal(out, self.aggregate_direct_mean, check_dtype=False)

    def test_aggregate_direct_sum(self):
        """Test daily mean summation."""
        out = tstoolbox.aggregate(
            statistic="sum", groupby="daily", input_ts="tests/data_flat.csv"
        ).astype("Float64")
        assert_frame_equal(out, self.aggregate_direct_sum, check_dtype=False)

    def test_aggregate_cli_mean(self):
        """Test CLI mean, daily (by default) aggregation."""
        args = (
            "tstoolbox aggregate "
            '--statistic="mean" '
            '--input_ts="tests/data_flat.csv"'
        )
        args = shlex.split(args)
        out = pd.read_csv(
            BytesIO(subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0])
        )
        assert_frame_equal(
            out, pd.read_csv(BytesIO(self.aggregate_cli_mean)), check_dtype=False
        )

    def test_aggregate_cli_sum(self):
        """Test CLI summation, daily (by default) aggregation."""
        args = (
            "tstoolbox aggregate "
            '--statistic="sum" '
            '--input_ts="tests/data_flat.csv"'
        )
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()[0]
        self.assertEqual(out, self.aggregate_cli_sum)


def test_aggregate_groupby():
    """Test API ValueError, groupby and agg_interval."""
    with pytest.raises(ValueError):
        _ = tstoolbox.aggregate(
            statistic="sum",
            groupby="D",
            agg_interval="D",
            input_ts="tests/data_flat.csv",
        )


def test_aggregate_ninterval_groupby():
    """Test API ValueError, ninterval and groupby."""
    with pytest.raises(ValueError):
        _ = tstoolbox.aggregate(
            statistic="sum", groupby="7D", ninterval=7, input_ts="tests/data_flat.csv"
        )


def test_aggregate_bad_statistic():
    """Test API statistic name."""
    with pytest.raises(ValueError):
        _ = tstoolbox.aggregate(
            statistic="camel", groupby="D", input_ts="tests/data_flat.csv"
        )


def test_aggregate_agg_interval():
    """Test API agg_interval."""
    with pytest.warns(UserWarning):
        _ = tstoolbox.aggregate(
            statistic="mean", agg_interval="D", input_ts="tests/data_flat.csv"
        )


def test_aggregate_ninterval():
    """Test API ninterval."""
    with pytest.warns(UserWarning):
        _ = tstoolbox.aggregate(
            statistic="mean", ninterval=7, input_ts="tests/data_flat.csv"
        )
