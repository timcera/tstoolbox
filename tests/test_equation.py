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


class TestEquation(TestCase):
    def setUp(self):
        dindex = pd.date_range('2000-01-01T00:00:00', periods=2, freq='D')
        ts1 = pd.TimeSeries([4.5, 4.6], index=dindex)
        ts2 = pd.TimeSeries([20.0, 20.4], index=dindex)
        self.equation = pd.DataFrame({'Value': ts1, 'Value_equation': ts2},
                index=dindex, dtype='float32')
        self.equation.index.name = 'Datetime'

        self.equation_cli = capture(tsutils._printiso, self.equation)

        dindex = pd.date_range('2000-01-01T00:00:00', periods=6, freq='D')
        ts1 = [4.50, 4.60, 4.70, 4.60, 4.50, 4.40]
        ts2 = [45.60, 90.50, 34.20, 23.10, 7.20, 4.30]
        ts3 = [-4.00389911875, -5.61841686236, -5.87322876716, -4.86614401542, -2.6934178416, -4.60800663972]
        self.equation_multiple_cols_01 = pd.DataFrame({'Value':ts1,
            'Value1':ts2, '__equation':ts3}, index=dindex, dtype='float32')
        self.equation_multiple_cols_01.index.name = 'Datetime'

        self.equation_result = pd.DataFrame({'__equation':pd.np.array(ts2)*10},
            index=dindex, dtype='float32')

        self.equation_multiple_cols_01_cli = capture(tsutils._printiso, self.equation_multiple_cols_01)

        ts3 = [50.1, 95.1, 38.9, 27.7, 11.7, 8.7]
        self.equation_multiple_cols_02 = pd.DataFrame({'Value':ts1,
            'Value1':ts2, '__equation':ts3}, index=dindex, dtype='float32')
        self.equation_multiple_cols_02.index.name = 'Datetime'

        self.equation_multiple_cols_02_cli = capture(tsutils._printiso, self.equation_multiple_cols_02)

        ts3 = [0, 97.92, 41.66, 30.52, 14.46, 0]
        ts3[0] = pd.np.nan
        ts3[-1] = pd.np.nan
        self.equation_multiple_cols_03 = pd.DataFrame({'Value':ts1,
            'Value1':ts2, '__equation':ts3}, index=dindex, dtype='float32')
        self.equation_multiple_cols_03.index.name = 'Datetime'

        self.equation_multiple_cols_03_cli = capture(tsutils._printiso,
                self.equation_multiple_cols_03, float_format='%.2f')

        dindex = pd.date_range('2011-01-01T00:00:00', periods=48, freq='H')
        ts1 = [2.0]*48
        ts2 = [5.2]*48
        ts2[0] = pd.np.nan
        ts2[-1] = pd.np.nan
        self.equation_multiple_cols_04 = pd.DataFrame({'Value': ts1,
            'Value_equation': ts2}, index=dindex, dtype='float32')
        self.equation_multiple_cols_04.index.name = 'Datetime'

        self.equation_multiple_cols_04_cli = capture(tsutils._printiso, self.equation_multiple_cols_04)

    def test_equation(self):
        out = tstoolbox.equation('x*4 + 2', input_ts='tests/data_simple.csv', print_input=True)
        assert_frame_equal(out, self.equation)

    def test_equation_multiple_cols_01(self):
        out = tstoolbox.equation('sin(x1)*4 + cos(x2)*2', input_ts='tests/data_multiple_cols.csv', print_input=True)
        assert_frame_equal(out, self.equation_multiple_cols_01)

    def test_equation_multiple_cols_02(self):
        out = tstoolbox.equation('x1 + x2', input_ts='tests/data_multiple_cols.csv', print_input=True)
        assert_frame_equal(out, self.equation_multiple_cols_02)

    def test_equation_multiple_cols_03(self):
        out = tstoolbox.equation('x1[t] + 0.6*maximum(x1[t-1], x1[t+1]) + x2', input_ts='tests/data_multiple_cols.csv', print_input=True)
        assert_frame_equal(out, self.equation_multiple_cols_03)

    def test_equation_multiple_cols_04(self):
        out = tstoolbox.equation('x[t] + 0.6*maximum(x[t-1], x[t+1]) + x[t]', input_ts='tests/data_flat.csv', print_input=True)
        self.maxDiff = None
        assert_frame_equal(out, self.equation_multiple_cols_04)

    def test_equation_multiple_cols_05(self):
        # Almost same as test_equation_multiple_cols_04
        out = tstoolbox.equation('x + 0.6*maximum(x[t-1], x[t+1]) + x[t]', input_ts='tests/data_flat.csv', print_input=True)
        self.maxDiff = None
        assert_frame_equal(out, self.equation_multiple_cols_04)

    def test_equation_cols_over_nine(self):
        input_ts = tstoolbox.read('tests/data_multiple_cols.csv,tests/data_multiple_cols.csv,tests/data_multiple_cols.csv,tests/data_multiple_cols.csv,tests/data_multiple_cols.csv')
        out = tstoolbox.equation('x10*10', input_ts=input_ts)
        self.maxDiff = None
        assert_frame_equal(out, self.equation_result)

    def test_equation_cli(self):
        args = 'tstoolbox equation "x*4 + 2" --input_ts="tests/data_simple.csv"  --print_input=True'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate()[0]
        self.maxDiff = None
        self.assertEqual(out, self.equation_cli)

    def test_equation_multiple_cols_01_cli(self):
        args = 'tstoolbox equation "sin(x1)*4 + cos(x2)*2" --input_ts="tests/data_multiple_cols.csv" --print_input=True'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate()[0]
        self.maxDiff = None
        self.assertEqual(out, self.equation_multiple_cols_01_cli)

    def test_equation_multiple_cols_02_cli(self):
        args = 'tstoolbox equation "x1 + x2" --input_ts="tests/data_multiple_cols.csv" --print_input=True'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate()[0]
        self.maxDiff = None
        self.assertEqual(out, self.equation_multiple_cols_02_cli)

    def test_equation_multiple_cols_03_cli(self):
        args = 'tstoolbox equation "x1[t] + 0.6*maximum(x1[t-1], x1[t+1]) + x2" --input_ts="tests/data_multiple_cols.csv" --print_input=True --float_format="%.2f"'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate()[0]
        self.maxDiff = None
        self.assertEqual(out, self.equation_multiple_cols_03_cli)

    def test_equation_multiple_cols_04_cli(self):
        args = 'tstoolbox equation "x[t] + 0.6*maximum(x[t-1], x[t+1]) + x[t]" --input_ts="tests/data_flat.csv" --print_input=True'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate()[0]
        self.maxDiff = None
        self.assertEqual(out, self.equation_multiple_cols_04_cli)

    def test_equation_multiple_cols_05_cli(self):
        # Almost same as test_equation_multiple_cols_04_cli
        args = 'tstoolbox equation "x + 0.6*maximum(x[t-1], x[t+1]) + x[t]" --input_ts="tests/data_flat.csv" --print_input=True'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE, stdin=subprocess.PIPE).communicate()[0]
        self.maxDiff = None
        self.assertEqual(out, self.equation_multiple_cols_04_cli)
