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
def ewm_window(input_ts='-',
               columns=None,
               start_date=None,
               end_date=None,
               dropna='no',
               skiprows=None,
               index_type='datetime',
               names=None,
               clean=False,
               statistic='',
               alpha_com=None,
               alpha_span=None,
               alpha_halflife=None,
               alpha=None,
               min_periods=0,
               adjust=True,
               ignore_na=False,
               source_units=None,
               target_units=None,
               print_input=False,
               ):
    """Calculate exponential weighted functions.

    Exactly one of center of mass, span, half-life, and alpha must be provided.
    Allowed values and relationship between the parameters are specified in the
    parameter descriptions above; see the link at the end of this section for
    a detailed explanation.

    When adjust is True (default), weighted averages are calculated using
    weights (1-alpha)**(n-1), (1-alpha)**(n-2), . . . , 1-alpha, 1.

    When adjust is False, weighted averages are calculated recursively as:
        weighted_average[0] = arg[0]; weighted_average[i]
        = (1-alpha)*weighted_average[i-1] + alpha*arg[i].

    When ignore_na is False (default), weights are based on absolute positions.
    For example, the weights of x and y used in calculating the final weighted
    average of [x, None, y] are (1-alpha)**2 and 1 (if adjust is True), and
    (1-alpha)**2 and alpha (if adjust is False).

    When ignore_na is True (reproducing pre-0.15.0 behavior), weights are based
    on relative positions. For example, the weights of x and y used in
    calculating the final weighted average of [x, None, y] are 1-alpha and
    1 (if adjust is True), and 1-alpha and alpha (if adjust is False).

    More details can be found at
    http://pandas.pydata.org/pandas-docs/stable/computation.html#exponentially-weighted-windows

    Parameters
    ----------
    statistic : str
        Statistic applied to each window.

        +------+--------------------+
        | corr | correlation        |
        +------+--------------------+
        | cov  | covariance         |
        +------+--------------------+
        | mean | mean               |
        +------+--------------------+
        | std  | standard deviation |
        +------+--------------------+
        | var  | variance           |
        +------+--------------------+

    alpha_com : float, optional
        Specify decay in terms of center of mass, alpha=1/(1+com), for com>=0

    alpha_span : float, optional
        Specify decay in terms of span, alpha=2/(span+1), for span1

    alpha_halflife : float, optional
        Specify decay in terms of half-life, alpha=1-exp(log(0.5)/halflife),
        for halflife>0

    alpha : float, optional
        Specify smoothing factor alpha directly, 0<alpha<=1

    min_periods : int, default 0
        Minimum number of observations in window required to have a value
        (otherwise result is NA).

    adjust : boolean, default True
        Divide by decaying adjustment factor in beginning periods to account
        for imbalance in relative weightings (viewing EWMA as a moving average)

    ignore_na : boolean, default False
        Ignore missing values when calculating weights.

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

    ntsd = tsd.ewm(alpha_com=alpha_com,
                   alpha_span=alpha_span,
                   alpha_halflife=alpha_halflife,
                   alpha=alpha,
                   min_periods=min_periods,
                   adjust=adjust,
                   ignore_na=ignore_na,
                   )

    if statistic:
        ntsd = eval('ntsd.{0}()'.format(statistic))

    return tsutils.print_input(print_input,
                               tsd,
                               ntsd,
                               '_ewm_' + statistic)
