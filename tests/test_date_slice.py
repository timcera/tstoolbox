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


class TestDate_slice(TestCase):

    def test_convert(self):
        out = capture(tstoolbox.date_slice, input_ts='tests/test_aggregate.csv', start_date='2011-01-01T12:00:00', end_date='2011-01-01T14:00:00')
        self.assertEqual(out,
"""Datetime, Value
2011-01-01 12:00:00 ,  2.0
2011-01-01 13:00:00 ,  2.0
2011-01-01 14:00:00 ,  2.0
""")
