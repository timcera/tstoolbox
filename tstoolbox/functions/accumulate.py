#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils

warnings.filterwarnings('ignore')


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def accumulate(input_ts='-',
               columns=None,
               start_date=None,
               end_date=None,
               dropna='no',
               clean=False,
               statistic='sum',
               round_index=None,
               skiprows=None,
               index_type='datetime',
               names=None,
               source_units=None,
               target_units=None,
               print_input=False):
    """Calculate accumulating statistics.

    Parameters
    ----------
    statistic : str
        [optional, default is 'sum']
        'sum', 'max', 'min', 'prod'
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {clean}
    {round_index}
    {skiprows}
    {index_type}
    {names}
    {print_input}
    {source_units}
    {target_units}

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
    try:
        ntsd = eval('tsd.cum{0}()'.format(statistic))
    except AttributeError:
        raise ValueError("""
*
*   Statistic {0} is not implemented.
*
""".format(statistic))
    return tsutils.print_input(print_input, tsd, ntsd, '_' + statistic)
