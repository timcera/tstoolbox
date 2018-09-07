#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_convert_index
----------------------------------

Tests for `tstoolbox.convert_index`

"""

from unittest import TestCase
from pandas.util.testing import assert_frame_equal
import shlex
import subprocess

import pandas as pd

from tstoolbox import tstoolbox
from tstoolbox import tsutils


class Testconvert_index(TestCase):
    def setUp(self):
        self.read_direct = pd.read_csv('tests/data_sunspot_index.csv',
                                       index_col=0)
        self.read_direct = tsutils.memory_optimize(self.read_direct)

    def test_convert_index_from_UTC(self):
        ''' Test
        '''
        out = tstoolbox.convert_index('number',
                                      input_ts='tests/data_sunspot.csv',
                                      interval='M',
                                      epoch='1850-01-01')
        out.index.name = '1850-01-01_date'
        assert_frame_equal(out, self.read_direct)
