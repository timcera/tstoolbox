#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_createts
----------------------------------

Tests for `tstoolbox` module.
"""

from pandas.util.testing import TestCase
from pandas.util.testing import assert_frame_equal, assert_index_equal
import shlex
import subprocess

import pandas as pd

from tstoolbox import tstoolbox
from tstoolbox import tsutils


class TestRead(TestCase):

    def setUp(self):
        ''' Prepare in-memory versions of the files ./data_flat.csv
        '''
        self.data = pd.DataFrame.from_csv('tests/data_flat.csv')
        self.data.index.name = 'Datetime'

    def test_createts_from_input(self):
        ''' Create a ts of data_flat.csv
        '''
        out = tstoolbox.createts(input_ts=self.data).index
        out1 = tstoolbox.read('tests/data_flat.csv').index
        print(out)
        print(out1)
        assert_index_equal(out, out1)

    def test_unstack_from_dates(self):
        ''' create a ts file from start/end dates and freq
        '''
        sdate = self.data.index[0]
        edate = self.data.index[-1]
        freq = tsutils.asbestfreq(self.data)
        freq = freq.index.freqstr
        out = tstoolbox.createts(start_date=sdate, end_date=edate, freq=freq).index
        out.name = 'Datetime'
        assert_index_equal(out, self.data.index)
