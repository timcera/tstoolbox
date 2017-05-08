#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_describe
----------------------------------

Tests for `tstoolbox` module.
"""

import shlex
import subprocess
from unittest import TestCase
from pandas.util.testing import assert_frame_equal

from . import capture

import pandas as pd

from tstoolbox import tstoolbox
import tstoolbox.tsutils as tsutils


class TestDescribe(TestCase):

    def setUp(self):
        index = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']
        self.date_slice = pd.DataFrame([1672.000000, 836.905383,
                                        843.901292, 0.000000, 158.425000, 578.800000, 1253.450000,
                                        4902.000000], index=index, columns=['Area'])
        self.date_slice.index.name = 'Statistic'
        self.date_slice_cli = capture.capture(tsutils._printiso,
                                              self.date_slice,
                                              force_print_index=True)
        self.date_slice.index.name = 'UniqueID'

    def test_describe(self):
        ''' Test of describe API.
        '''
        out = tstoolbox.describe(input_ts='tests/data_sunspot.csv')
        out.index.name = 'UniqueID'
        assert_frame_equal(out, self.date_slice)

    def test_describe_cli(self):
        ''' Test of describe CLI.
        '''
        args = 'tstoolbox describe --input_ts="tests/data_sunspot.csv"'
        args = shlex.split(args)
        out = subprocess.Popen(args,
                               stdout=subprocess.PIPE,
                               stdin=subprocess.PIPE).communicate()[0]
        self.assertEqual(out, self.date_slice_cli)
