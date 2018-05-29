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

from .functions.plot import plot
from .functions.createts import createts
from .functions.filter import filter
from .functions.read import read
from .functions.date_slice import date_slice
from .functions.describe import describe
from .functions.peak_detection import peak_detection
from .functions.convert import convert
from .functions.equation import equation
from .functions.pick import pick
from .functions.stdtozrxp import stdtozrxp
from .functions.tstopickle import tstopickle
from .functions.accumulate import accumulate
from .functions.ewm_window import ewm_window
from .functions.expanding_window import expanding_window
from .functions.rolling_window import rolling_window
from .functions.aggregate import aggregate
from .functions.replace import replace
from .functions.clip import clip
from .functions.add_trend import add_trend
from .functions.remove_trend import remove_trend
from .functions.calculate_fdc import calculate_fdc
from .functions.stack import stack
from .functions.unstack import unstack
from .functions.fill import fill
from .functions.gof import gof
from .functions.dtw import dtw
from .functions.pca import pca
from .functions.normalization import normalization
from .functions.converttz import converttz
from .functions.convert_index_to_julian import convert_index_to_julian
from .functions.pct_change import pct_change
from .functions.rank import rank
from .functions.date_offset import date_offset

warnings.filterwarnings('ignore')


@mando.command()
def about():
    """Display version number and system information."""
    tsutils.about(__name__)


def main():
    """Main function."""
    if not os.path.exists('debug_tstoolbox'):
        sys.tracebacklimit = 0
    mando.main()


if __name__ == '__main__':
    main()
