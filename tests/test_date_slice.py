# -*- coding: utf-8 -*-

import shlex
import subprocess
from unittest import TestCase

import pandas as pd
from pandas.testing import assert_frame_equal

from tstoolbox import tstoolbox, tsutils

from . import capture


class TestDate_slice(TestCase):
    def setUp(self):
        dr = pd.date_range("2011-01-01T12:00:00", periods=3, freq="H")
        self.date_slice = pd.DataFrame([2, 2, 2], index=dr, columns=["Value"])
        self.date_slice = tsutils.memory_optimize(self.date_slice)

        self.date_slice_cli = capture.capture(tsutils.printiso, self.date_slice)

    def test_date_slice(self):
        """Test date_slice API."""
        out = tstoolbox.date_slice(
            input_ts="tests/data_flat.csv",
            start_date="2011-01-01T12:00:00",
            end_date="2011-01-01T14:00:00",
        )
        assert_frame_equal(out, self.date_slice)

    def test_date_slice_cli(self):
        """Test date_slice CLI."""
        args = (
            "tstoolbox date_slice "
            '--input_ts="tests/data_flat.csv" '
            '--start_date="2011-01-01T12:00:00" '
            '--end_date="2011-01-01T14:00:00"'
        )
        args = shlex.split(args)
        out = subprocess.Popen(
            args, stdout=subprocess.PIPE, stdin=subprocess.PIPE
        ).communicate()[0]
        self.assertEqual(out, self.date_slice_cli)
