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
def calculate_fdc(input_ts='-',
                  columns=None,
                  start_date=None,
                  end_date=None,
                  clean=False,
                  skiprows=None,
                  index_type='datetime',
                  names=None,
                  percent_point_function=None,
                  plotting_position='weibull',
                  source_units=None,
                  target_units=None,
                  ascending=True):
    """Return the frequency distribution curve.

    DOES NOT return a time-series.

    Parameters
    ----------
    percent_point_function : str
        [optional, default is None]

        The distribution used to shift the plotting position values.
        Choose from 'norm', 'lognorm', 'weibull', and None.
    plotting_position : str
        [optional, default is 'weibull']

        {plotting_position_table}

    ascending : bool
        Sort order defaults to True.
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

    ppf = tsutils._set_ppf(percent_point_function)
    newts = pd.DataFrame()
    for col in tsd:
        tmptsd = tsd[col].dropna()
        xdat = ppf(tsutils._set_plotting_position(tmptsd.count(),
                                                  plotting_position)) * 100
        tmptsd.sort_values(ascending=ascending, inplace=True)
        tmptsd.index = xdat
        newts = newts.join(tmptsd, how='outer')
    newts.index.name = 'Plotting_position'
    newts = newts.groupby(newts.index).first()
    return tsutils.printiso(newts,
                            showindex='always')
