"""A correlation routine."""

from typing import List, Literal

import numpy as np
import pandas as pd
from pydantic import conint, validate_arguments
from toolbox_utils import tsutils

from . import lag


def autocorrelation(series):
    """Perform autocorrelation if lags==0"""
    n = len(series)
    data = np.asarray(series)
    mean = np.mean(data)
    c0 = np.sum((data - mean) ** 2) / float(n)

    def r(h):
        return ((data[: n - h] - mean) * (data[h:] - mean)).sum() / float(n) / c0

    x = np.arange(n) + 1
    y = [r(loc) for loc in x]
    return y


@tsutils.transform_args(lags=tsutils.make_list)
@validate_arguments
@tsutils.doc(tsutils.docstrings)
def correlation(
    lags: List[conint(ge=0)],
    method: Literal["pearson", "kendall", "spearman"] = "pearson",
    input_ts="-",
    start_date=None,
    end_date=None,
    columns=None,
    clean=False,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    skiprows=None,
    round_index=None,
    dropna=None,
):
    """Develop a correlation between time-series and potentially lags.

    Parameters
    ----------
    lags : str, int, or list
        If lags are greater than 0 then returns a cross correlation matrix
        between all time-series and all lags.  If an integer will calculate and
        use all lags up to and including the lag number.  If a list will
        calculate each lag in the list.  If a string must be a comma separated
        list of integers.

        If lags == 0 then will return an auto-correlation on each input
        time-series.

        Python example::

            lags=[2, 5, 3]

        Command line example::

            --lags='2,5,3'

    method : str
        [optional, default to "pearson"]

        Method of correlation::

            pearson : standard correlation coefficient

            kendall : Kendall Tau correlation coefficient

            spearman : Spearman rank correlation

    ${input_ts}

    ${start_date}

    ${end_date}

    ${clean}

    ${skiprows}

    ${index_type}

    ${names}

    ${source_units}

    ${target_units}

    ${columns}

    ${tablefmt}

    ${round_index}

    ${dropna}
    """
    tsd = tsutils.common_kwds(
        input_ts,
        skiprows=skiprows,
        names=names,
        index_type=index_type,
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        source_units=source_units,
        target_units=target_units,
        clean=clean,
    )

    if len(lags) == 1 and int(lags[0]) == 0:
        ntsd = pd.DataFrame()
        for cname, cdata in tsd.iteritems():
            y = autocorrelation(cdata)
            ntsd = ntsd.join(
                pd.DataFrame(y, index=tsd.index, columns=[cname]), how="outer"
            )
        try:
            x = pd.timedelta_range(start=1, end=len(ntsd) + 1, freq=tsd.index.freqstr)
        except ValueError:
            x = np.arange(len(ntsd)) + 1
        ntsd.index = x
        return ntsd

    ntsd = lag.lag(
        lags,
        input_ts=tsd,
    )
    return ntsd.corr(method=method)
