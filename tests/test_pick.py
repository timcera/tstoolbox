#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_pick
----------------------------------

Tests for `tstoolbox` module.
"""

from unittest import TestCase
import sys
import shlex
import subprocess
try:
    from cStringIO import StringIO
except:
    from io import StringIO

import pandas

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


class TestPick(TestCase):
    def setUp(self):
        dr = pandas.date_range('2000-01-01', periods=6, freq='D')

        ts1 = pandas.TimeSeries([ 4.5,  4.6,  4.7,  4.6,  4.5,  4.4],
                index=dr)
        ts2 = pandas.TimeSeries([45.6, 90.5, 34.2, 23.1,  7.2,  4.3],
                index=dr)
        self.pick_multiple_direct = pandas.DataFrame(ts2,
                columns=['Value1_0'])
        self.pick_multiple_direct = self.pick_multiple_direct.join(pandas.DataFrame(ts1, columns=['Value_1']))

        self.pick_cli = capture(tsutils._printiso, self.pick_multiple_direct)

    def test_pick(self):
        out = tstoolbox.pick('2,1', 'tests/test_multiple_cols.csv')
        self.assertEqual(out, self.pick_multiple_direct)

    def test_pick_cli(self):
        args = 'tstoolbox pick 2,1 "tests/test_multiple_cols.csv"'
        args = shlex.split(args)
        out = subprocess.Popen(args, stdout=subprocess.PIPE).communicate()
        self.assertEqual(out[0], self.pick_cli)
