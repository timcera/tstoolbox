#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import warnings

import mando

from . import tsutils
from .functions.accumulate import accumulate
from .functions.add_trend import add_trend
from .functions.aggregate import aggregate
from .functions.calculate_fdc import calculate_fdc
from .functions.calculate_kde import calculate_kde
from .functions.clip import clip
from .functions.convert import convert
from .functions.convert_index import convert_index
from .functions.convert_index_to_julian import convert_index_to_julian
from .functions.converttz import converttz
from .functions.createts import createts
from .functions.date_offset import date_offset
from .functions.date_slice import date_slice
from .functions.describe import describe
from .functions.dtw import dtw
from .functions.equation import equation
from .functions.ewm_window import ewm_window
from .functions.expanding_window import expanding_window
from .functions.fill import fill
from .functions.filter import filter
from .functions.gof import gof
from .functions.normalization import normalization
from .functions.pca import pca
from .functions.pct_change import pct_change
from .functions.peak_detection import peak_detection
from .functions.pick import pick
from .functions.plot import plot
from .functions.rank import rank
from .functions.read import read
from .functions.remove_trend import remove_trend
from .functions.replace import replace
from .functions.rolling_window import rolling_window
from .functions.stack import stack
from .functions.stdtozrxp import stdtozrxp
from .functions.tstopickle import tstopickle
from .functions.unstack import unstack

warnings.filterwarnings('ignore')


@mando.command()
def about():
    """Display version number and system information."""
    tsutils.about(__name__)


def main():
    """Set debug and run mando.main function."""
    if not os.path.exists('debug_tstoolbox'):
        sys.tracebacklimit = 0
    mando.main()


if __name__ == '__main__':
    main()
