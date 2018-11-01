#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from unittest import TestCase

from tstoolbox import tstoolbox


class TestRead(TestCase):

    def setUp(self):
        """Prepare in-memory versions of the files ./data_flat.csv."""
        self.fdata = tstoolbox.read('tests/data_flat.csv,tests/data_sunspot.csv')
        self.fdata.index.name = 'Datetime'

    def test_creation(self):
        _ = tstoolbox.calculate_fdc(input_ts=self.fdata)
