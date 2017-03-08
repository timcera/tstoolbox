#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_stack_unstack
----------------------------------

Tests for `tstoolbox` module.
"""

from pandas.util.testing import TestCase
from pandas.util.testing import assert_frame_equal
import shlex
import subprocess

import pandas as pd

from tstoolbox import tstoolbox
from tstoolbox import tsutils


class TestRead(TestCase):
    def setUp(self):
        ''' Prepare in-memory versions of the files data_stacked.csv and
            data_unstacked.csv.
        '''
        self.stacked = pd.read_csv('tests/data_stacked.csv',
                                             index_col=0,
                                             parse_dates=True)
        self.stacked.index.name = 'Datetime'
        self.stacked = tsutils.memory_optimize(self.stacked)

        self.unstacked = pd.read_csv('tests/data_unstacked.csv',
                                               index_col=0,
                                               parse_dates=True)
        self.unstacked.rename(columns=lambda x: x.strip('\'\" '))
        self.unstacked.index.name = 'Datetime'
        self.unstacked = tsutils.memory_optimize(self.unstacked)

    def test_stack(self):
        ''' Stack the data_unstacked.csv file and compare against the
            in-memory version of the data_stacked.csv file.
        '''
        out = tstoolbox.stack(input_ts='tests/data_unstacked.csv')
        assert_frame_equal(out, self.stacked)

    def test_unstack(self):
        ''' Unstack the data_stacked.csv file and compare against the
            in-memory version of the data_unstacked.csv file.
        '''
        out = tstoolbox.unstack('Columns', input_ts='tests/data_stacked.csv')
        assert_frame_equal(out, self.unstacked)
