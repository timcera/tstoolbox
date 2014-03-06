#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_tstoolbox
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


class TestFill(TestCase):
    def setUp(self):
        dindex = pd.date_range('2011-01-01T00:00:00', periods=26, freq='H')
        self.ats = pd.np.ones((26))*2
        self.ats = pd.DataFrame(self.ats, index=dindex, columns=['Value_fill'])
        self.ats.index.name = 'Datetime'

        self.ats_cli = capture(tsutils._printiso, self.ats)

        self.ffill_compare = self.ats.copy()
        self.ffill_compare['Value_fill']['2011-01-01T09:00:00':'2011-01-01T12:00:00'] = 3
        self.ffill_compare['Value_fill']['2011-01-01T13:00:00'] = 9

        self.ffill_compare_cli = capture(tsutils._printiso, self.ffill_compare)

        self.bfill_compare = self.ats.copy()
        self.bfill_compare['Value_fill']['2011-01-01T09:00:00'] = 3
        self.bfill_compare['Value_fill']['2011-01-01T10:00:00':'2011-01-01T13:00:00'] = 9

        self.bfill_compare_cli = capture(tsutils._printiso, self.bfill_compare)

        self.linear_compare = self.ats.copy()
        self.linear_compare['Value_fill']['2011-01-01T09:00:00'] = 3.0
        self.linear_compare['Value_fill']['2011-01-01T10:00:00'] = 4.5
        self.linear_compare['Value_fill']['2011-01-01T11:00:00'] = 6.0
        self.linear_compare['Value_fill']['2011-01-01T12:00:00'] = 7.5
        self.linear_compare['Value_fill']['2011-01-01T13:00:00'] = 9.0

        self.linear_compare_cli = capture(tsutils._printiso, self.linear_compare)

        self.nearest_compare = self.ats.copy()
        self.nearest_compare['Value_fill']['2011-01-01T09:00:00':'2011-01-01T11:00:00'] = 3.0
        self.nearest_compare['Value_fill']['2011-01-01T12:00:00':'2011-01-01T13:00:00'] = 9.0

        self.nearest_compare_cli = capture(tsutils._printiso, self.nearest_compare)

        self.mean_compare = self.ats.copy()
        self.mean_compare['Value_fill']['2011-01-01T01:00:00'] = 2.4210526315789473
        self.mean_compare['Value_fill']['2011-01-01T09:00:00'] = 3.0
        self.mean_compare['Value_fill']['2011-01-01T10:00:00'] = 2.4210526315789473
        self.mean_compare['Value_fill']['2011-01-01T11:00:00'] = 2.4210526315789473
        self.mean_compare['Value_fill']['2011-01-01T12:00:00'] = 2.4210526315789473
        self.mean_compare['Value_fill']['2011-01-01T13:00:00'] = 9.0
        self.mean_compare['Value_fill']['2011-01-01T16:00:00'] = 2.4210526315789473
        self.mean_compare['Value_fill']['2011-01-01T22:00:00'] = 2.4210526315789473
        self.mean_compare['Value_fill']['2011-01-01T23:00:00'] = 2.4210526315789473

        self.mean_compare_cli = capture(tsutils._printiso, self.mean_compare)

    def test_fill_ffill_direct(self):
        out = tstoolbox.fill(input_ts='tests/data_missing.csv')
        self.maxDiff = None
        assert_frame_equal(out, self.ffill_compare)

    def test_fill_bfill(self):
        out = tstoolbox.fill(method='bfill', input_ts='tests/data_missing.csv')
        self.maxDiff = None
        assert_frame_equal(out, self.bfill_compare)

    def test_fill_linear(self):
        out = tstoolbox.fill(method='linear', input_ts='tests/data_missing.csv')
        self.maxDiff = None
        assert_frame_equal(out, self.linear_compare)

    def test_fill_nearest(self):
        out = tstoolbox.fill(method='nearest', input_ts='tests/data_missing.csv')
        self.maxDiff = None
        assert_frame_equal(out, self.nearest_compare)

    def test_fill_mean(self):
        out = tstoolbox.fill(method='mean', input_ts='tests/data_missing.csv')
        self.maxDiff = None
        assert_frame_equal(out, self.mean_compare)

    def test_fill_ffill_cli(self):
        args = 'tstoolbox fill --input_ts=tests/data_missing.csv'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate(input=self.ats_cli)[0]
        self.maxDiff = None
        self.assertEqual(out, self.ffill_compare_cli)

    def test_fill_bfill_cli(self):
        args = 'tstoolbox fill --method="bfill" --input_ts=tests/data_missing.csv'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate(input=self.ats_cli)[0]
        self.maxDiff = None
        self.assertEqual(out, self.bfill_compare_cli)

    def test_fill_linear_cli(self):
        args = 'tstoolbox fill --method="linear" --input_ts=tests/data_missing.csv'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate(input=self.ats_cli)[0]
        self.maxDiff = None
        self.assertEqual(out, self.linear_compare_cli)

    def test_fill_nearest_cli(self):
        args = 'tstoolbox fill --method="nearest" --input_ts=tests/data_missing.csv'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate(input=self.ats_cli)[0]
        self.maxDiff = None
        self.assertEqual(out, self.nearest_compare_cli)

    def test_fill_mean_cli(self):
        args = 'tstoolbox fill --method="mean" --input_ts=tests/data_missing.csv'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate(input=self.ats_cli)[0]
        self.maxDiff = None
        self.assertEqual(out, self.mean_compare_cli)
