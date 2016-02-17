
from __future__ import print_function

import os
import tempfile

import numpy as np
import matplotlib
from matplotlib.testing.compare import compare_images
from pandas.util.testing import TestCase

from tstoolbox import tstoolbox


class TestPlot(TestCase):

    def setUp(self):
        self.df = tstoolbox.read('tests/data_sine.csv')
        fp, self.fname = tempfile.mkstemp(suffix='.png')

    def test_sine(self):
        plt = tstoolbox.plot(input_ts=self.df, ofilename=None)
        plt.savefig(self.fname)

        # different versions of matplotlib have slightly different fonts so I
        # set the tolerance pretty high to account for this problem.
        results = compare_images('tests/baseline_images/test_plot/sine.png',
                                 self.fname,
                                 10)
        if results is None:
            return True
        base, ext = os.path.splitext(self.fname)
        os.remove('%s-%s%s' % (base, 'failed-diff', ext))
        print(results)
        assert False

    def tearDown(self):
        ''' Remove the temporary files.
        '''
        os.remove(self.fname)
