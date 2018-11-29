#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import pandas as pd

from .. import tsutils


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def pct_change(input_ts='-',
               columns=None,
               start_date=None,
               end_date=None,
               dropna='no',
               skiprows=None,
               index_type='datetime',
               names=None,
               clean=False,
               periods=1,
               fill_method='pad',
               limit=None,
               freq=None,
               print_input=False,
               round_index=None,
               source_units=None,
               target_units=None,
               float_format='%g'):
    """Return the percent change between times.

    Parameters
    ----------
    periods : int
        [optional, default is 1]

        The number of intervals to calculate percent change across.
    fill_method : str
        [optional, defaults to 'pad']

        Fill method for NA.  Defaults to 'pad'.
    limit
        [optional, defaults to None]

        Is the minimum number of consecutive NA values where no more filling
        will be made.
    freq : str
        [optional, defaults to None]

        A pandas time offset string to represent the interval.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {skiprows}
    {index_type}
    {names}
    {clean}
    {source_units}
    {target_units}
    {print_input}
    {float_format}
    {round_index}

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

    # Trying to save some memory
    if print_input:
        otsd = tsd.copy()
    else:
        otsd = pd.DataFrame()

    return tsutils.print_input(print_input,
                               otsd,
                               tsd.pct_change(periods=periods,
                                              fill_method=fill_method,
                                              limit=limit,
                                              freq=freq),
                               '_pct_change',
                               float_format=float_format)
