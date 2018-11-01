#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import TestCase

import pandas as pd
from pandas.util.testing import assert_frame_equal

from tstoolbox import tstoolbox
from tstoolbox import tsutils


class Testconvert_index(TestCase):
    def setUp(self):
        self.read_direct = pd.read_csv('tests/data_sunspot_index.csv',
                                       index_col=0)
        self.read_direct = tsutils.memory_optimize(self.read_direct)

    def test_convert_index_from_UTC(self):
        out = tstoolbox.convert_index('number',
                                      input_ts='tests/data_sunspot.csv',
                                      interval='M',
                                      epoch='1850-01-01')
        out.index.name = '1850-01-01_date'
        assert_frame_equal(out, self.read_direct)
