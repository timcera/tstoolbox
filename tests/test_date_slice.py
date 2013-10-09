#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_tstoolbox
----------------------------------

Tests for `tstoolbox` module.
"""

import shlex
import subprocess
from unittest import TestCase
import sys
try:
    from cStringIO import StringIO
except:
    from io import StringIO

import pandas as pd

import tstoolbox
import tsutils


def capture(func, *args, **kwds):
    sys.stdout = StringIO()      # capture output
    out = func(*args, **kwds)
    out = sys.stdout.getvalue()  # release output
    try:
        out = bytes(out, 'utf-8')
    except:
        pass
    return out


class TestDate_slice(TestCase):
    def setUp(self):
        dr = pd.date_range('2011-01-01T12:00:00', periods=3, freq='H')
        self.date_slice = pd.DataFrame([2.0, 2.0, 2.0], index=dr, columns=['Value'])
        self.date_slice_cli = capture(tsutils._printiso, self.date_slice)

    def test_date_slice(self):
        out = tstoolbox.date_slice(input_ts='tests/test_aggregate.csv', start_date='2011-01-01T12:00:00', end_date='2011-01-01T14:00:00')
        self.assertEqual(out, self.date_slice)

    def test_date_slice_cli(self):
        args = 'tstoolbox date_slice --input_ts="tests/test_aggregate.csv" --start_date="2011-01-01T12:00:00" --end_date="2011-01-01T14:00:00"'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate()[0]
        self.assertEqual(out, self.date_slice_cli)
