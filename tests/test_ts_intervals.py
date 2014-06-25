#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_ts_intervals
----------------------------------

Tests for `tstoolbox` module.
"""

import sys
import shlex
import subprocess
import os
import glob
try:
    from cStringIO import StringIO
except:
    from io import StringIO

import pandas

from pandas.util.testing import TestCase
from pandas.util.testing import assert_frame_equal
import pandas as pd

from tstoolbox import tstoolbox

pandacodes = ['A', 'AS', 'BA', 'BAS',    # Annual
              'Q', 'QS', 'BQ', 'BQS',    # Quarterly
              'M', 'MS', 'BM', 'BMS',    # Monthly
              'W',                       # Weekly
              'D', 'B',                  # Daily
              'H', 'T', 'S', 'L', 'U']   # Intra-daily

pd_tstep_minterval = {
                      'A': ('1800-01-01', 300, 10),
                      'AS': ('1800-01-01', 300, 10),
                      'BA': ('1800-01-01', 300, 10),
                      'BAS': ('1800-01-01', 300, 10),
                      'Q': ('1900-01-01', 300, 12),
                      'QS': ('1900-01-01', 300, 12),
                      'BQ': ('1900-01-01', 300, 12),
                      'BQS': ('1900-01-01', 300, 12),
                      'M': ('1900-01-01', 240, 12),
                      'MS': ('1900-01-01', 240, 12),
                      'BM': ('1900-01-01', 240, 12),
                      'BMS': ('1900-01-01', 240, 12),
                      'W': ('2000-01-01', 240, 12),
                      'D': ('2000-01-01', 3660, 10),
                      'B': ('2000-01-01', 3660, 10),
                      'H': ('2000-01-01', 24*366*5, 5),
                      'T': ('2000-01-01', 60*24*366, 2),
                      'S': ('2000-01-01', 60*60*24*2, 3),
                      'L': ('2000-01-01', 1000, 2),
                      'U': ('2000-01-01', 1000, 2),
                     }

def capture(func, *args, **kwds):
    sys.stdout = StringIO()      # capture output
    out = func(*args, **kwds)
    out = sys.stdout.getvalue()  # release output
    return out


class TestAddTrend(TestCase):
    def setUp(self):

        for testpc in pandacodes:
            sdate, periods, nintervals = pd_tstep_minterval[testpc]
            for tstep in range(1, nintervals):
                aperiods = periods//tstep
                dr = pd.date_range(sdate, periods=aperiods, freq='{0}{1}'.format(tstep,testpc))
                df = pd.DataFrame(pd.np.arange(aperiods), index=dr)
                df.index.name = 'index'
                df.columns = ['data']
                df.to_csv(os.path.join('tests',
                                       'tmptestdata',
                                       'datafile_{0}_{1}.csv'.format(
                                           tstep,
                                           testpc)), sep=',', header=True)


    def test_ts_intervals(self):
        fnames = glob.glob(os.path.join('tests',
                                        'tmptestdata',
                                        '*.csv'))
        matches = {
                   '4Q': '1A',
                   '8Q': '2A',
                   '4BQS': '1BAS',
                   '8BQS': '2BAS',
                   '4BQ': '1BA',
                   '8BQ': '2BA',
                   '4QS': '1AS',
                   '8QS': '2AS',
                   '2M': '1M',
                   '3M': '1Q',
                   '4M': '1M',
                   '5M': '1M',
                   '6M': '2Q',
                   '7M': '1M',
                   '8M': '1M',
                   '9M': '3Q',
                   '10M': '1M',
                   '11M': '1M',
                   '2BMS': '1BMS',
                   '3BMS': '1BQS',
                   '4BMS': '1BMS',
                   '5BMS': '1BMS',
                   '6BMS': '2BQS',
                   '7BMS': '1BMS',
                   '8BMS': '1BMS',
                   '9BMS': '3BQS',
                   '10BMS': '1BMS',
                   '11BMS': '1BMS',
                   '2BM': '1BM',
                   '3BM': '1BQ',
                   '4BM': '1BM',
                   '5BM': '1BM',
                   '6BM': '2BQ',
                   '7BM': '1BM',
                   '8BM': '1BM',
                   '9BM': '3BQ',
                   '10BM': '1BM',
                   '11BM': '1BM',
                   '2MS': '1MS',
                   '3MS': '1QS',
                   '4MS': '1MS',
                   '5MS': '1MS',
                   '6MS': '2QS',
                   '7MS': '1MS',
                   '8MS': '1MS',
                   '9MS': '3QS',
                   '10MS': '1MS',
                   '11MS': '1MS',
                   '5B': '1W',
                   '7D': '1W',
                   }

        for fname in fnames:
            df = tstoolbox.read(fname)
            inferred_code = df.index.inferred_freq
            if inferred_code is None:
                continue
            tinterval = os.path.split(fname)[1]
            tinterval = os.path.splitext(tinterval)[0]
            tinterval, tcode = tinterval.split('_')[1:3]
            testcode = '{0}{1}'.format(tinterval, tcode)
            if inferred_code[0] not in '123456789':
                inferred_code = '1' + inferred_code
            if testcode in matches:
                testcode = matches[testcode]
            self.assertEqual(testcode, inferred_code.split('-')[0])

    def tearDown(self):
        fnames = glob.glob(os.path.join('tests',
                                        'tmptestdata',
                                        '*.csv'))
        for fname in fnames:
            if os.path.exists(fname):
                os.remove(fname)
