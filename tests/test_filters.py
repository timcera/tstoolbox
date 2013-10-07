#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_peak_detect
----------------------------------

Tests for `tstoolbox` module.
"""

from unittest import TestCase
import sys
try:
    from cStringIO import StringIO
except:
    from io import StringIO

import pandas as pd

import tstoolbox


def capture(func, *args, **kwds):
    sys.stdout = StringIO()      # capture output
    out = func(*args, **kwds)
    out = sys.stdout.getvalue()  # release output
    return out


test_sinwave = 'Datetime, 0, 0_peak, 0_valley\n2000-01-01 00:00:00 ,  0.0,  ,  \n2000-01-01 01:00:00 ,  0.258819045103,  ,  \n2000-01-01 02:00:00 ,  0.5,  ,  \n2000-01-01 03:00:00 ,  0.707106781187,  ,  \n2000-01-01 04:00:00 ,  0.866025403784,  ,  \n2000-01-01 05:00:00 ,  0.965925826289,  ,  \n2000-01-01 06:00:00 ,  1.0, 1.0,  \n2000-01-01 07:00:00 ,  0.965925826289,  ,  \n2000-01-01 08:00:00 ,  0.866025403784,  ,  \n2000-01-01 09:00:00 ,  0.707106781187,  ,  \n2000-01-01 10:00:00 ,  0.5,  ,  \n2000-01-01 11:00:00 ,  0.258819045103,  ,  \n2000-01-01 12:00:00 ,  1.22464679915e-16,  ,  \n2000-01-01 13:00:00 ,  -0.258819045103,  ,  \n2000-01-01 14:00:00 ,  -0.5,  ,  \n2000-01-01 15:00:00 ,  -0.707106781187,  ,  \n2000-01-01 16:00:00 ,  -0.866025403784,  ,  \n2000-01-01 17:00:00 ,  -0.965925826289,  ,  \n2000-01-01 18:00:00 ,  -1.0,  , -1.0\n2000-01-01 19:00:00 ,  -0.965925826289,  ,  \n2000-01-01 20:00:00 ,  -0.866025403784,  ,  \n2000-01-01 21:00:00 ,  -0.707106781187,  ,  \n2000-01-01 22:00:00 ,  -0.5,  ,  \n2000-01-01 23:00:00 ,  -0.258819045103,  ,  \n'

class TestFilter(TestCase):
    def setUp(self):
        dindex = pd.date_range('2000-01-01T00:00:00', periods=24, freq='H')
        self.ats = pd.np.arange(0, 360, 15)
        self.ats = pd.np.sin(2*pd.np.pi*self.ats/360)
        self.ats = pd.DataFrame(self.ats, index=dindex)

#    def test_filter_flat(self):
#        out = capture(tstoolbox.filter, 'flat', input_ts=self.ats, print_input=True)
#        self.maxDiff = None
#        self.assertEqual(out, test_sinwave)
#
#    def test_filter_hanning(self):
#        out = capture(tstoolbox.filter, 'hanning', input_ts=self.ats, print_input=True)
#        self.maxDiff = None
#        self.assertEqual(out, test_sinwave)
