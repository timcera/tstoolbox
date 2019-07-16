#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unittest import TestCase

from tstoolbox import tstoolbox


class Testgof(TestCase):
    def setUp(self):
        self.df = tstoolbox.read(['tests/data_sunspot.csv',
                                  'tests/data_sunspot.csv'])
        self.gof = [['Bias', 0.0],
                    ['Percent bias', 0.0],
                    ['Absolute percent bias', 0.0],
                    ['Root-mean-square Deviation (RMSD)', 0.0],
                    ['Centered RMSD (CRMSD)', 0.0],
                    ['Correlation coefficient (r)', 1.0],
                    ['Skill score (Murphy)', 1.0],
                    ['Nash-Sutcliffe Efficiency', 1.0],
                    ['Kling-Gupta Efficiency', 1.0],
                    ['Index of agreement', 1.0],
                    ['Common count observed and simulated', 1672],
                    ['Count of NaNs', '', 0, 0]]

    def test_gof(self):
        """Test of gof API."""
        out = tstoolbox.gof(input_ts=self.df, stats = ['bias',
                                                       'pc_bias',
                                                       'apc_bias',
                                                       'rmsd',
                                                       'crmsd',
                                                       'corrcoef',
                                                       'murphyss',
                                                       'nse',
                                                       'kge',
                                                       'index_agreement',
                                                       'brierss',])
        assert out == self.gof
