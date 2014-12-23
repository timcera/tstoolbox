#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_normalization
----------------------------------

Tests for `tstoolbox` module.
"""

import shlex
import subprocess
from pandas.util.testing import TestCase
from pandas.util.testing import assert_frame_equal
import sys
try:
    from cStringIO import StringIO
except:
    from io import StringIO

import pandas as pd

from tstoolbox import tstoolbox
import tstoolbox.tsutils as tsutils


def capture(func, *args, **kwds):
    sys.stdout = StringIO()      # capture output
    out = func(*args, **kwds)
    out = sys.stdout.getvalue()  # release output
    try:
        out = bytes(out, 'utf-8')
    except:
        pass
    return out


class TestDescribe(TestCase):
    def setUp(self):
        self.data_0_to_1 = tstoolbox.read('tests/data_sunspot_normalized_0_to_1.csv')
        self.data_10_to_20 = tstoolbox.read('tests/data_sunspot_normalized_10_to_20.csv')
        self.data_zscore = tstoolbox.read('tests/data_sunspot_normalized_zscore.csv')

    def test_normalize_0_to_1(self):
        out = tstoolbox.normalization(input_ts='tests/data_sunspot.csv')
        assert_frame_equal(out, self.data_0_to_1)

    def test_normalize_10_to_20(self):
        out = tstoolbox.normalization(min_limit=10, max_limit=20, input_ts='tests/data_sunspot.csv')
        assert_frame_equal(out, self.data_10_to_20)

    def test_normalize(self):
        out = tstoolbox.normalization(mode='zscore', input_ts='tests/data_sunspot.csv')
        assert_frame_equal(out, self.data_zscore)

