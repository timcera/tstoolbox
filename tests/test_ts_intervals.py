import logging
import os
import tempfile
from unittest import TestCase

import numpy as np
import pandas as pd

from tstoolbox import tstoolbox

if float(".".join(pd.__version__.split(".")[:2])) >= 2.2:
    pandacodes = [
        "YE",
        "YS",
        "BYE",
        "BYS",  # Annual
        "QE",
        "QS",
        "BQE",
        "BQS",  # Quarterly
        "ME",
        "MS",
        "BME",
        "BMS",  # Monthly
        "W",  # Weekly
        "D",
        "B",  # Daily
        "h",
        "min",
        "s",
        "ms",
        "us",
    ]  # Intra-daily

    pd_tstep_minterval = {
        "YE": ("1800-01-01", 300, 10),
        "YS": ("1800-01-01", 300, 10),
        "BYE": ("1800-01-01", 300, 10),
        "BYS": ("1800-01-01", 300, 10),
        "QE": ("1900-01-01", 300, 12),
        "QS": ("1900-01-01", 300, 12),
        "BQE": ("1900-01-01", 300, 12),
        "BQS": ("1900-01-01", 300, 12),
        "ME": ("1900-01-01", 240, 12),
        "MS": ("1900-01-01", 240, 12),
        "BME": ("1900-01-01", 240, 12),
        "BMS": ("1900-01-01", 240, 12),
        "W": ("2000-01-01", 240, 12),
        "D": ("2000-01-01", 3660, 10),
        "B": ("2000-01-03", 3660, 10),
        "h": ("2000-01-01", 24 * 366 * 5, 5),
        "min": ("2000-01-01", 60 * 24 * 366, 2),
        "s": ("2000-01-01", 60 * 60 * 24 * 2, 3),
        "ms": ("2000-01-01", 1000, 2),
        "us": ("2000-01-01", 1000, 2),
    }
else:
    pandacodes = [
        "A",
        "AS",
        "BA",
        "BAS",  # Annual
        "Q",
        "QS",
        "BQ",
        "BQS",  # Quarterly
        "M",
        "MS",
        "BM",
        "BMS",  # Monthly
        "W",  # Weekly
        "D",
        "B",  # Daily
        "H",
        "T",
        "S",
        "L",
        "U",
    ]  # Intra-daily

    pd_tstep_minterval = {
        "A": ("1800-01-01", 300, 10),
        "AS": ("1800-01-01", 300, 10),
        "BA": ("1800-01-01", 300, 10),
        "BAS": ("1800-01-01", 300, 10),
        "Q": ("1900-01-01", 300, 12),
        "QS": ("1900-01-01", 300, 12),
        "BQ": ("1900-01-01", 300, 12),
        "BQS": ("1900-01-01", 300, 12),
        "M": ("1900-01-01", 240, 12),
        "MS": ("1900-01-01", 240, 12),
        "BM": ("1900-01-01", 240, 12),
        "BMS": ("1900-01-01", 240, 12),
        "W": ("2000-01-01", 240, 12),
        "D": ("2000-01-01", 3660, 10),
        "B": ("2000-01-03", 3660, 10),
        "H": ("2000-01-01", 24 * 366 * 5, 5),
        "T": ("2000-01-01", 60 * 24 * 366, 2),
        "S": ("2000-01-01", 60 * 60 * 24 * 2, 3),
        "L": ("2000-01-01", 1000, 2),
        "U": ("2000-01-01", 1000, 2),
    }


logging.basicConfig(filename="example.log", encoding="utf-8", level=logging.DEBUG)


class TestAddTrend(TestCase):
    def setUp(self):
        """Create a whole bunch of time-series at different intervals.

        Uses 'pandacodes' and different number of intervals, the maximum number
        of intervals is the third term in each entry in pd_tstep_minterval.
        Then writes them out to files that will be read in the tests.
        """
        self.fps = {}
        for testpc in pandacodes:
            sdate, periods, nintervals = pd_tstep_minterval[testpc]
            for tstep in range(1, nintervals):
                aperiods = periods // tstep
                dr = pd.date_range(sdate, periods=aperiods, freq=f"{tstep}{testpc}")
                df = pd.DataFrame(np.arange(aperiods), index=dr)
                df.index.name = "index"
                df.columns = ["data"]
                self.fps[(tstep, testpc)] = tempfile.mkstemp()
                df.to_csv(self.fps[(tstep, testpc)][1], sep=",", header=True)

    def test_ts_intervals(self):
        """Test many intervals to make sure tstoolbox makes a good guess."""
        # 'matches' lists out things that should match but tstoolbox will give
        # the simpler answer.  For example 4 quarters is equal to a 1 year
        # interval.  Also difficult for tstoolbox to figure out that a 2 month
        # interval isn't a 1 month interval with gaps.
        if float(".".join(pd.__version__.split(".")[:2])) > 2.0:
            matches = {
                "4BQS": "1BYS",
                "8BQS": "2BYS",
                "1A": "1YE",
                "2A": "2YE",
                "3A": "3YE",
                "4A": "4YE",
                "5A": "5YE",
                "6A": "6YE",
                "7A": "7YE",
                "8A": "8YE",
                "9A": "9YE",
                "1AS": "1YS",
                "2AS": "2YS",
                "3AS": "3YS",
                "4AS": "4YS",
                "5AS": "5YS",
                "6AS": "6YS",
                "7AS": "7YS",
                "8AS": "8YS",
                "9AS": "9YS",
                "1BA": "1BYE",
                "2BA": "2BYE",
                "3BA": "3BYE",
                "4BA": "4BYE",
                "5BA": "5BYE",
                "6BA": "6BYE",
                "7BA": "7BYE",
                "8BA": "8BYE",
                "9BA": "9BYE",
                "1BAS": "1BYS",
                "2BAS": "2BYS",
                "3BAS": "3BYS",
                "4BAS": "4BYS",
                "5BAS": "5BYS",
                "6BAS": "6BYS",
                "7BAS": "7BYS",
                "8BAS": "8BYS",
                "9BAS": "9BYS",
                "1Q": "1QE",
                "2Q": "2QE",
                "3Q": "3QE",
                "4Q": "1YE",
                "5Q": "5QE",
                "6Q": "6QE",
                "7Q": "7QE",
                "8Q": "2YE",
                "9Q": "9QE",
                "10Q": "10QE",
                "11Q": "11QE",
                "4QS": "1YS",
                "8QS": "2YS",
                "1BQ": "1BQE",
                "2BQ": "2BQE",
                "3BQ": "3BQE",
                "4BQ": "1BYE",
                "5BQ": "5BQE",
                "6BQ": "6BQE",
                "7BQ": "7BQE",
                "8BQ": "2BYE",
                "9BQ": "9BQE",
                "10BQ": "10BQE",
                "11BQ": "11BQE",
                "1M": "1ME",
                "2M": "2ME",
                "3M": "1QE",
                "4M": "4ME",
                "5M": "5ME",
                "6M": "2QE",
                "7M": "7ME",
                "8M": "8ME",
                "9M": "3QE",
                "3BMS": "1BQS",
                "6BMS": "2BQS",
                "9BMS": "3BQS",
                "3BME": "1BQE",
                "6BME": "2BQE",
                "9BME": "3BQE",
                "3ME": "1QE",
                "6ME": "2QE",
                "9ME": "3QE",
                "3MS": "1QS",
                "6MS": "2QS",
                "9MS": "3QS",
                "10M": "10ME",
                "11M": "11ME",
                "1BM": "1BME",
                "2BM": "2BME",
                "3BM": "1BQE",
                "4BM": "4BME",
                "5BM": "5BME",
                "6BM": "2BQE",
                "7BM": "7BME",
                "8BM": "8BME",
                "9BM": "3BQE",
                "10BM": "10BME",
                "11BM": "11BME",
                "1H": "1h",
                "2H": "2h",
                "3H": "3h",
                "4H": "4h",
                "5H": "5h",
                "6H": "6h",
                "7H": "7h",
                "8H": "8h",
                "9H": "9h",
                "1T": "1min",
                "1S": "1s",
                "2S": "2s",
                "1L": "1ms",
                "1U": "1us",
                "7D": "1W",
                "2B": "2D",
                "3B": "1D",
                "4B": "2D",
                "5B": "1W",
                "6B": "2D",
                "7B": "1D",
                "8B": "2D",
                "9B": "1D",
                "4QE": "1YE",
                "8QE": "2YE",
                "4BQE": "1BYE",
                "8BQE": "2BYE",
            }
        else:
            matches = {
                "4Q": "1A",
                "8Q": "2A",
                "4BQS": "1BAS",
                "8BQS": "2BAS",
                "4BQ": "1BA",
                "8BQ": "2BA",
                "4QS": "1AS",
                "8QS": "2AS",
                "2M": "1M",
                "3M": "1Q",
                "4M": "1M",
                "5M": "1M",
                "6M": "2Q",
                "7M": "1M",
                "8M": "1M",
                "9M": "3Q",
                "10M": "1M",
                "11M": "1M",
                "2BMS": "1BMS",
                "3BMS": "1BQS",
                "4BMS": "1BMS",
                "5BMS": "1BMS",
                "6BMS": "2BQS",
                "7BMS": "1BMS",
                "8BMS": "1BMS",
                "9BMS": "3BQS",
                "10BMS": "1BMS",
                "11BMS": "1BMS",
                "2BM": "1BM",
                "3BM": "1BQ",
                "4BM": "1BM",
                "5BM": "1BM",
                "6BM": "2BQ",
                "7BM": "1BM",
                "8BM": "1BM",
                "9BM": "3BQ",
                "10BM": "1BM",
                "11BM": "1BM",
                "2MS": "1MS",
                "3MS": "1QS",
                "4MS": "1MS",
                "5MS": "1MS",
                "6MS": "2QS",
                "7MS": "1MS",
                "8MS": "1MS",
                "9MS": "3QS",
                "10MS": "1MS",
                "11MS": "1MS",
                "5B": "1W",
                "7D": "1W",
                # BUG!!!
                "2B": "2D",
                "3B": "1D",
                "4B": "2D",
                "6B": "2D",
                "7B": "1D",
                "8B": "2D",
                "9B": "1D",
            }

        for key in self.fps:
            df = tstoolbox.read(self.fps[key][1])
            inferred_code = df.index.inferred_freq
            if inferred_code is None:
                continue
            testcode = "{}{}".format(*key)
            if inferred_code[0] not in "123456789":
                inferred_code = "1" + inferred_code
            testcode = matches.get(testcode, testcode)
            logging.warning(f"{testcode} {inferred_code}")
            icode = inferred_code.split("-")[0]
            try:
                self.assertEqual(testcode, icode)
            except AssertionError:
                self.assertEqual(
                    [i for i in testcode if not i.isdigit()],
                    [i for i in icode if not i.isdigit()],
                )

    def tearDown(self):
        """Remove the temporary files."""
        for _, fname in self.fps.items():
            if os.path.exists(fname[1]):
                os.remove(fname[1])
