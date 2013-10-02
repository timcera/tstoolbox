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


class TestConvert(TestCase):

    def test_convert(self):
        out = capture(tstoolbox.convert, input_ts='tests/test.csv')
        self.assertEqual(out,
"""Datetime, Value_convert
2000-01-01 00:00:00 ,  4.5
2000-01-02 00:00:00 ,  4.6
""")

    def test_convert_2_3(self):
        out = capture(tstoolbox.convert, input_ts='tests/test.csv', factor=2, offset=2)
        self.assertEqual(out,
"""Datetime, Value_convert
2000-01-01 00:00:00 ,  11.0
2000-01-02 00:00:00 ,  11.2
""")

