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
def date_offset(intervals,
                offset,
                columns=None,
                dropna='no',
                clean=False,
                skiprows=None,
                index_type='datetime',
                names=None,
                input_ts='-',
                start_date=None,
                end_date=None,
                source_units=None,
                target_units=None,
                round_index=None):
    """Apply an offset to a time-series.

    Parameters
    ----------
    intervals: int

        Number of intervals of `offset` to shift the time index.  A positive
        integer moves the index forward, negative moves it backwards.

    offset: str

        Pandas offset.

        {pandas_offset_codes}

    {input_ts}
    {start_date}
    {end_date}
    {columns}
    {round_index}
    {dropna}
    {clean}
    {skiprows}
    {index_type}
    {source_units}
    {target_units}
    {names}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts,
                                                  skiprows=skiprows,
                                                  names=names,
                                                  index_type=index_type),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna='no',
                              source_units=source_units,
                              target_units=target_units,
                              clean=clean)

    return tsutils.printiso(tsd.tshift(intervals, offset),
                            showindex='always')
