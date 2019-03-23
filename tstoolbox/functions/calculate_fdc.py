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


@mando.command('calculate_fdc',
               formatter_class=RSTHelpFormatter,
               doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def calculate_fdc_cli(input_ts='-',
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
                      sort_values='ascending',
                      sort_index='ascending'):
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

    sort_values : str
        [optional, default is 'ascending']

        Sort order is either 'ascending' or 'descending'.

    sort_index : str
        [optional, default is 'ascending']

        Sort order is either 'ascending' or 'descending'.
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
    tsutils._printiso(calculate_fdc(input_ts=input_ts,
                                    columns=columns,
                                    start_date=start_date,
                                    end_date=end_date,
                                    clean=clean,
                                    skiprows=skiprows,
                                    index_type=index_type,
                                    names=names,
                                    percent_point_function=percent_point_function,
                                    plotting_position=plotting_position,
                                    source_units=source_units,
                                    target_units=target_units,
                                    sort_values=sort_values,
                                    sort_index=sort_index),
                      showindex='always')


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
                  sort_values='ascending',
                  sort_index='ascending'):
    """Return the frequency distribution curve."""
    if sort_values not in ['ascending', 'descending']:
        raise ValueError("""
*
*   The 'sort_values' option must be either 'ascending' or 'descending'.
*   You gave {0}.
*
""".format(sort_values))
    if sort_index not in ['ascending', 'descending']:
        raise ValueError("""
*
*   The 'sort_index' option must be either 'ascending' or 'descending'.
*   You gave {0}.
*
""".format(sort_index))
    sort_values = bool(sort_values == 'ascending')

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

    ppf = tsutils.set_ppf(percent_point_function)
    newts = pd.DataFrame()
    for col in tsd:
        tmptsd = tsd[col].dropna()
        xdat = ppf(tsutils.set_plotting_position(tmptsd.count(),
                                                 plotting_position)) * 100
        tmptsd.sort_values(ascending=sort_values, inplace=True)
        tmptsd.index = xdat
        newts = newts.join(tmptsd, how='outer')
    newts.index.name = 'Plotting_position'
    newts = newts.groupby(newts.index).first()
    if sort_index == 'descending':
        return newts.iloc[::-1]
    return newts
