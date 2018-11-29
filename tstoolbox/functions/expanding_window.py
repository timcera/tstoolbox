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
def expanding_window(input_ts='-',
                     columns=None,
                     start_date=None,
                     end_date=None,
                     dropna='no',
                     skiprows=None,
                     index_type='datetime',
                     names=None,
                     clean=False,
                     statistic='',
                     min_periods=1,
                     center=False,
                     source_units=None,
                     target_units=None,
                     print_input=False,
                     ):
    """Calculate an expanding window statistic.

    Parameters
    ----------
    statistic : str

        +-----------+----------------------+
        | statistic | Meaning              |
        +===========+======================+
        | corr      | correlation          |
        +-----------+----------------------+
        | count     | count of real values |
        +-----------+----------------------+
        | cov       | covariance           |
        +-----------+----------------------+
        | kurt      | kurtosis             |
        +-----------+----------------------+
        | max       | maximum              |
        +-----------+----------------------+
        | mean      | mean                 |
        +-----------+----------------------+
        | median    | median               |
        +-----------+----------------------+
        | min       | minimum              |
        +-----------+----------------------+
        | skew      | skew                 |
        +-----------+----------------------+
        | std       | standard deviation   |
        +-----------+----------------------+
        | sum       | sum                  |
        +-----------+----------------------+
        | var       | variance             |
        +-----------+----------------------+

    min_periods : int, default 1
        Minimum number of observations in window required to have a value

    center : boolean, default False
        Set the labels at the center of the window.

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

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts,
                                                  skiprows=skiprows,
                                                  names=names,
                                                  index_type=index_type),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              dropna=dropna,
                              source_units=source_units,
                              target_units=target_units,
                              clean=clean)

    ntsd = tsd.expanding(min_periods=min_periods,
                         center=center)

    if statistic:
        ntsd = eval('ntsd.{0}()'.format(statistic))

    return tsutils.print_input(print_input,
                               tsd,
                               ntsd,
                               '_expanding_' + statistic)
