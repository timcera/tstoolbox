#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_tstoolbox
----------------------------------

Tests for `tstoolbox` module.
"""

from unittest import TestCase
import sys
try:
    from cStringIO import StringIO
except:
    from io import StringIO

import tstoolbox


def capture(func, *args, **kwds):
    sys.stdout = StringIO()      # capture output
    out = func(*args, **kwds)
    out = sys.stdout.getvalue()  # release output
    return out


class TestRead(TestCase):

    def test_read(self):
        out = capture(tstoolbox.read, 'tests/test.csv')
        self.assertEqual(out,
"""Datetime, Value
2000-01-01 00:00:00 ,  4.5
2000-01-02 00:00:00 ,  4.6
""")

    def test_read_mulitple(self):
        out = capture(tstoolbox.read, 'tests/test.csv', 'tests/test.csv')
        self.assertEqual(out,
"""Datetime, test_Value, test_1_Value
2000-01-01 00:00:00 ,  4.5, 4.5
2000-01-02 00:00:00 ,  4.6, 4.6
""")


class TestAggregate(TestCase):

    def test_aggregate_mean(self):
        out = capture(tstoolbox.aggregate, statistic='mean', agg_interval='daily', input_ts='tests/test_aggregate.csv')
        self.assertEqual(out,
"""Datetime, Value_mean
2011-01-01 00:00:00 ,  2.0
2011-01-02 00:00:00 ,  2.0
""")

    def test_aggregate_sum(self):
        out = capture(tstoolbox.aggregate, statistic='sum', agg_interval='daily', input_ts='tests/test_aggregate.csv')
        self.assertEqual(out,
"""Datetime, Value_sum
2011-01-01 00:00:00 ,  48.0
2011-01-02 00:00:00 ,  48.0
""")

