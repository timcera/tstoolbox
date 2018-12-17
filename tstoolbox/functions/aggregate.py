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
              groupby=None,
              statistic='mean',
              columns=None,
              start_date=None,
              end_date=None,
              dropna='no',
              clean=False,
              agg_interval=None,
              ninterval=None,
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

        'mean', 'sem', 'sum', 'std', 'max', 'min', 'median', 'first', 'last' or
        'ohlc' to calculate on each group.
    {groupby}
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
    agg_interval :
        DEPRECATED:
        Use the 'groupby' option instead.
    ninterval :
        DEPRECATED:
        Just prefix the number in front of the 'groupby' pandas offset code.
    """
    statslist = ['mean',
                 'sum',
                 'std',
                 'sem',
                 'max',
                 'min',
                 'median',
                 'first',
                 'last',
                 'ohlc']
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

    if agg_interval is not None:
        if groupby is not None:
            raise ValueError("""
*
*   You cannot specify both 'groupby' and 'agg_interval'.  The 'agg_interval'
*   option is deprecated in favor of 'groupby'.
*
""")
        warnings.warn("""
*
*   The 'agg_interval' option has been deprecated in favor of 'groupby' to be
*   consistent with other tstoolbox commands.
*
""")
        groupby = aggd.get(agg_interval, agg_interval)
    else:
        groupby = 'D'

    if ninterval is not None:
        ninterval = int(ninterval)

        import re
        try:
            _ = int(re.match(r'^\d+', groupby).group())
            raise ValueError("""
*
*   You cannot specify the 'ninterval' option and prefix a number in the
*   'groupby' option.  The 'ninterval' option is deprecated in favor of
*   prefixing the number in the pandas offset code used in the 'groupby'
*   option.
*
""")
        except AttributeError:
            pass

        warnings.warn("""
*
*   The 'ninterval' option has been deprecated in favor of prefixing the desired
*   interval in front of the 'groupby' pandas offset code.
*
*   For example: instead of 'grouby="D"' and 'ninterval=7', you can just
*   have 'groupby="7D"'.
*
""")
    else:
        ninterval = ''

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
    methods = tsutils.make_list(statistic)
    newts = pd.DataFrame()
    for method in methods:
        tmptsd = eval("""tsd.resample('{0}{1}').{2}()""".format(ninterval,
                                                                groupby,
                                                                method))
        tmptsd.rename(columns=lambda x: x + '_' + method, inplace=True)
        newts = newts.join(tmptsd, how='outer')
    return tsutils.print_input(print_input, tsd, newts, '')
