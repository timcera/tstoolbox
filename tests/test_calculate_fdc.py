#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_createts
----------------------------------

Tests for `tstoolbox` module.
"""

from __future__ import print_function
import shlex
import subprocess

from pandas.util.testing import TestCase
from pandas.util.testing import assert_frame_equal, assert_index_equal, assertRaisesRegexp
import pandas as pd

from tstoolbox import tstoolbox
from tstoolbox import tsutils


class TestRead(TestCase):

    def setUp(self):
        ''' Prepare in-memory versions of the files ./data_flat.csv
        '''
        self.fdata = tstoolbox.read('tests/data_flat.csv,tests/data_sunspot.csv')
        self.fdata.index.name = 'Datetime'

    def test_creation(self):
        ''' Create
        '''
        out = tstoolbox.calculate_fdc(input_ts=self.fdata)

