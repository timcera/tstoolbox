# -*- coding: utf-8 -*-
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings
from typing import List, Optional

import mando
import pandas as pd
import typic
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal


warnings.filterwarnings("ignore")


@mando.command("ewm_window", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def ewm_window_cli(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    statistic="",
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
    tablefmt="csv",
):
    """Calculate exponential weighted functions.

    Exactly one of `alpha_com` (center of mass), `alpha_span`,
    `alpha_halflife`, and `alpha` must be provided to calculate the
    'alpha' term.  Allowed values and relationship between the
    parameters are specified in the parameter descriptions below; see
    the link at the end of this section for a detailed explanation.

    When `adjust` is True (default), weighted averages are calculated
    using weights `(1-alpha)**(n-1)`, `(1-alpha)**(n-2)`, . . . , `1-alpha`,
    1.

    When `adjust` is False, weighted averages are calculated recursively
    as: weighted_average[0] = arg[0]; weighted_average[i]
    = (1-alpha)*weighted_average[i-1] + alpha*arg[i].

    When `ignore_na` is False (default), weights are based on absolute
    positions.  For example, the weights of x and y used in calculating
    the final weighted average of [x, None, y] are (1-alpha)**2 and
    1 (if `adjust` is True), and (1-alpha)**2 and alpha (if `adjust` is
    False).

    When `ignore_na` is True weights are based on relative positions.
    For example, the weights of x and y used in calculating the final
    weighted average of [x, None, y] are 1-alpha and 1 (if `adjust` is
    True), and 1-alpha and alpha (if adjust is `False`).

    More details can be found at
    http://pandas.pydata.org/pandas-docs/stable/computation.html#exponentially-weighted-windows

    Parameters
    ----------
    statistic : str
        [optional, defaults to '']

        Statistic applied to each window.

        +-----------+--------------------+
        | statistic | Description        |
        +===========+====================+
        | corr      | correlation        |
        +-----------+--------------------+
        | cov       | covariance         |
        +-----------+--------------------+
        | mean      | mean               |
        +-----------+--------------------+
        | std       | standard deviation |
        +-----------+--------------------+
        | var       | variance           |
        +-----------+--------------------+
    alpha_com : float
        [optional, defaults to None]

        Specify decay in terms of center of mass::

            alpha = 1/(1+`alpha_com`), for `alpha_com` >= 0
    alpha_span : float
        [optional, defaults to None]

        Specify decay in terms of span::

            alpha = 2/(`alpha_span`+1), for `alpha_span` > 1
    alpha_halflife : float
        [optional, defaults to None]

        Specify decay in terms of half-life::

            alpha = 1-exp(log(0.5)/alpha_halflife), for
            alpha_halflife > 0
    alpha : float
        [optional, defaults to None]

        Specify smoothing factor "alpha" directly, 0<`alpha`<=1
    min_periods : int
        [optional, default is 0]

        Minimum number of observations in window required to have a value
        (otherwise result is NA).
    adjust : boolean
        [optional, default is True]

        Divide by decaying adjustment factor in beginning periods to account
        for imbalance in relative weightings (viewing EWMA as a moving average)
    ignore_na : boolean
        [optional, default is False]
        Ignore missing values when calculating weights.
    ${input_ts}
    ${columns}
    ${start_date}
    ${end_date}
    ${dropna}
    ${skiprows}
    ${index_type}
    ${names}
    ${clean}
    ${source_units}
    ${target_units}
    ${print_input}
    ${tablefmt}
    """
    tsutils.printiso(
        ewm_window(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            clean=clean,
            statistic=statistic,
            alpha_com=alpha_com,
            alpha_span=alpha_span,
            alpha_halflife=alpha_halflife,
            alpha=alpha,
            min_periods=min_periods,
            adjust=adjust,
            ignore_na=ignore_na,
            source_units=source_units,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


@tsutils.transform_args(statistic=tsutils.make_list)
@typic.al
@tsutils.copy_doc(ewm_window_cli)
def ewm_window(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    statistic: List[Optional[Literal["corr", "cov", "mean", "std", "var"]]] = "mean",
    alpha_com: Optional[tsutils.FloatGreaterEqualToZero] = None,
    alpha_span: Optional[tsutils.FloatGreaterEqualToOne] = None,
    alpha_halflife: Optional[tsutils.FloatGreaterEqualToZero] = None,
    alpha: Optional[tsutils.FloatBetweenZeroAndOneInclusive] = None,
    min_periods: Optional[tsutils.IntGreaterEqualToZero] = 0,
    adjust: bool = True,
    ignore_na: bool = False,
    source_units=None,
    target_units=None,
    print_input=False,
):
    """Calculate exponential weighted functions."""
    tsd = tsutils.common_kwds(
        input_ts,
        skiprows=skiprows,
        names=names,
        index_type=index_type,
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        dropna=dropna,
        source_units=source_units,
        target_units=target_units,
        clean=clean,
    )

    ntsd = tsd.ewm(
        alpha_com=alpha_com,
        alpha_span=alpha_span,
        alpha_halflife=alpha_halflife,
        alpha=alpha,
        min_periods=min_periods,
        adjust=adjust,
        ignore_na=ignore_na,
    )

    if statistic:
        nntsd = pd.DataFrame()
        for stat in statistic:
            ntsd = eval("ntsd.{}()".format(stat))
            ntsd = [tsutils.renamer(i, "ewm.{}".format(stat)) for i in ntsd.columns]
            nntsd = nntsd.join(ntsd, how="outer")
    else:
        nntsd = ntsd

    return tsutils.return_input(print_input, tsd, nntsd)
