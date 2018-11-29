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
def aggregate(input_ts='-',
              columns=None,
              start_date=None,
              end_date=None,
              dropna='no',
              clean=False,
              statistic='mean',
              agg_interval='D',
              ninterval=1,
              round_index=None,
              skiprows=None,
              index_type='datetime',
              names=None,
              source_units=None,
              target_units=None,
              print_input=False):
    """Take a time series and aggregate to specified frequency.

    Parameters
    ----------
    statistic : str
        [optional, defaults to 'mean']

        'mean', 'sum', 'std', 'max', 'min', 'median', 'first', or 'last'
        to calculate the aggregation.  Can also be a comma separated list of
        statistic methods.
    agg_interval : str
        [optional, defaults to 'D']
        The interval to aggregate the time series.  Any of the PANDAS
        offset codes.

        {pandas_offset_codes}

        There are some deprecated aggregation interval names in
        tstoolbox, DON'T USE!

        +---------------+-----+
        | Instead of... | Use |
        +===============+=====+
        | hourly        | H   |
        +---------------+-----+
        | daily         | D   |
        +---------------+-----+
        | monthly       | M   |
        +---------------+-----+
        | yearly        | A   |
        +---------------+-----+

    ninterval : int
        [optional, defaults to 1]

        The number of agg_interval to use for the aggregation.
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
    {source_units}
    {target_units}
    {print_input}

    """
    statslist = ['mean',
                 'sum',
                 'std',
                 'max',
                 'min',
                 'median',
                 'first',
                 'last']
    assert statistic in statslist, """
***
*** The statistic option must be one of:
*** {1}
*** to calculate the aggregation.
*** You gave {0}.
***
""".format(statistic, statslist)

    aggd = {'hourly': 'H',
            'daily': 'D',
            'monthly': 'M',
            'yearly': 'A'}
    agg_interval = aggd.get(agg_interval, agg_interval)

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
    methods = statistic.split(',')
    newts = pd.DataFrame()
    for method in methods:
        if method == 'mean':
            tmptsd = tsd.resample('{0:d}{1}'.format(ninterval,
                                                    agg_interval)).mean()
        elif method == 'sum':
            tmptsd = tsd.resample('{0:d}{1}'.format(ninterval,
                                                    agg_interval)).sum()
        elif method == 'std':
            tmptsd = tsd.resample('{0:d}{1}'.format(ninterval,
                                                    agg_interval)).std()
        elif method == 'max':
            tmptsd = tsd.resample('{0:d}{1}'.format(ninterval,
                                                    agg_interval)).max()
        elif method == 'min':
            tmptsd = tsd.resample('{0:d}{1}'.format(ninterval,
                                                    agg_interval)).min()
        elif method == 'median':
            tmptsd = tsd.resample('{0:d}{1}'.format(ninterval,
                                                    agg_interval)).median()
        elif method == 'first':
            tmptsd = tsd.resample('{0:d}{1}'.format(ninterval,
                                                    agg_interval)).first()
        elif method == 'last':
            tmptsd = tsd.resample('{0:d}{1}'.format(ninterval,
                                                    agg_interval)).last()
        tmptsd.rename(columns=lambda x: x + '_' + method, inplace=True)
        newts = newts.join(tmptsd, how='outer')
    return tsutils.print_input(print_input, tsd, newts, '')
