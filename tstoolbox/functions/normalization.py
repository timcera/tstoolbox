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
def normalization(input_ts='-',
                  columns=None,
                  start_date=None,
                  end_date=None,
                  dropna='no',
                  skiprows=None,
                  index_type='datetime',
                  names=None,
                  clean=False,
                  mode='minmax',
                  min_limit=0,
                  max_limit=1,
                  pct_rank_method='average',
                  print_input=False,
                  round_index=None,
                  source_units=None,
                  target_units=None,
                  float_format='%g'):
    """Return the normalization of the time series.

    Parameters
    ----------
    mode : str
        [optional, default is 'minmax']

        minmax
            min_limit +
            (X-Xmin)/(Xmax-Xmin)*(max_limit-min_limit)

        zscore
            X-mean(X)/stddev(X)

        pct_rank
            rank(X)*100/N
    min_limit : float
        [optional, defaults to 0]

        Defines the minimum limit of the minmax normalization.
    max_limit : float
        [optional, defaults to 1]

        Defines the maximum limit of the minmax normalization.
    pct_rank_method : str
        [optional, defaults to 'average']

        Defines how tied ranks are broken.  Can be 'average', 'min', 'max',
        'first', 'dense'.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {skiprows}
    {index_type}
    {names}
    {clean}
    {print_input}
    {float_format}
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

    # Trying to save some memory
    if print_input:
        otsd = tsd.copy()
    else:
        otsd = pd.DataFrame()

    if mode == 'minmax':
        tsd = (min_limit +
               (tsd - tsd.min()) /
               (tsd.max() - tsd.min()) *
               (max_limit - min_limit))
    elif mode == 'zscore':
        tsd = (tsd - tsd.mean()) / tsd.std()
    elif mode == 'pct_rank':
        tsd = tsd.rank(method=pct_rank_method, pct=True)
    else:
        raise ValueError("""
*
*   The 'mode' options are 'minmax', 'zscore', or 'pct_rank', you gave me
*   {0}.
*
""".format(mode))

    tsd = tsutils.memory_optimize(tsd)
    return tsutils.print_input(print_input,
                               otsd,
                               tsd,
                               '_{0}'.format(mode),
                               float_format=float_format)
