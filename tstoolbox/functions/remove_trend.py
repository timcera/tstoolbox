#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import pandas as pd

from .. import tsutils

warnings.filterwarnings('ignore')


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def remove_trend(input_ts='-',
                 columns=None,
                 start_date=None,
                 end_date=None,
                 dropna='no',
                 skiprows=None,
                 index_type='datetime',
                 names=None,
                 clean=False,
                 round_index=None,
                 source_units=None,
                 target_units=None,
                 print_input=False):
    """Remove a 'trend'.

    Parameters
    ----------
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {skiprows}
    {index_type}
    {names}
    {clean}
    {round_index}
    {source_units}
    {target_units}
    {print_input}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts,
                                                  skiprows=skiprows,
                                                  names=names,
                                                  index_type=index_type),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna,
                              source_units=source_units,
                              target_units=target_units,
                              clean=clean)
    ntsd = tsd.copy()
    for col in tsd.columns:
        index = tsd.index.astype('l')
        index = index - index[0]
        lin = pd.np.polyfit(index, tsd[col], 1)
        ntsd[col] = lin[0] * index + lin[1]
        ntsd[col] = tsd[col] - ntsd[col]
    return tsutils.print_input(
        print_input, tsd, ntsd, '_rem_trend')
