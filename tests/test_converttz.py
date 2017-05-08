#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_converttz
----------------------------------

Tests for `tstoolbox.converttz`

"""

from unittest import TestCase
from pandas.util.testing import assert_frame_equal
import shlex
import subprocess

import pandas as pd

from tstoolbox import tstoolbox
from tstoolbox import tsutils


class TestConvertTZ(TestCase):
    def setUp(self):
        self.read_direct = pd.read_csv('tests/data_sunspot_EST.csv',
                                       index_col=0,
                                       parse_dates=[0]).tz_localize('UTC').tz_convert('EST')
        self.read_direct = tsutils.memory_optimize(self.read_direct)

    def test_converttz_from_UTC(self):
        ''' Test
        '''
        out = tstoolbox.converttz('UTC',
                                  'EST',
                                  input_ts='tests/data_sunspot.csv')
        out.index.name = 'Datetime'
        assert_frame_equal(out, self.read_direct)
