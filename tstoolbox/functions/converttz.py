#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mando
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def converttz(fromtz,
              totz,
              input_ts='-',
              columns=None,
              start_date=None,
              end_date=None,
              round_index=None,
              dropna='no',
              clean=False,
              index_type='datetime',
              names=None,
              source_units=None,
              target_units=None,
              skiprows=None):
    """Convert the time zone of the index.

    Parameters
    ----------
    fromtz : str
        The time zone of the original time-series.
    totz : str
        The time zone of the converted time-series.
    {input_ts}
    {start_date}
    {end_date}
    {columns}
    {dropna}
    {clean}
    {skiprows}
    {index_type}
    {names}
    {source_units}
    {target_units}
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
    tsd = tsd.tz_localize(fromtz).tz_convert(totz)
    return tsutils.printiso(tsd,
                            showindex='always')
