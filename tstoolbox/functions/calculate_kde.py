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
def calculate_kde(ascending=True,
                  evaluate=False,
                  input_ts='-',
                  columns=None,
                  start_date=None,
                  end_date=None,
                  clean=False,
                  skiprows=None,
                  index_type='datetime',
                  source_units=None,
                  target_units=None,
                  names=None):
    """Return the kernel density estimation (KDE) curve.

    Returns a time-series or the KDE curve depending on the `evaluate` keyword.

    Parameters
    ----------
    ascending : bool
        Sort order defaults to True.
    evaluate : bool
        Whether or not to return a time-series of KDE density values or the KDE
        curve.  Defaults to False, which would return the KDE curve.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {skiprows}
    {index_type}
    {names}
    {source_units}
    {target_units}
    {clean}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts,
                                                  skiprows=skiprows,
                                                  names=names,
                                                  index_type=index_type),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              source_units=source_units,
                              target_units=target_units,
                              clean=clean)

    if len(tsd.columns) > 1:
        raise ValueError("""
*
*   Right now "calculate_kde" only support one time-series at a time.
*   You gave {0}.
*
""".format(tsd.columns))

    from scipy.stats import gaussian_kde

    tmptsd = tsd.dropna()
    ndf = tmptsd.sort_values(tmptsd.columns[0], ascending=ascending)
    gkde = gaussian_kde(ndf.iloc[:, 0])

    if evaluate is True:
        y = gkde.evaluate(tmptsd.iloc[:, 0])
        ndf = pd.DataFrame(y, index=tmptsd.index)
    else:
        y = gkde.evaluate(ndf.iloc[:, 0])
        ndf = pd.DataFrame(y)

    return tsutils.printiso(ndf,
                            showindex='always')
