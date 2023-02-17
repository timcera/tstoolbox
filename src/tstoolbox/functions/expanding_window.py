"""Collection of functions for the manipulation of time series."""

from typing import List, Literal, Optional

import pandas as pd
from pydantic import PositiveInt, validate_arguments
from toolbox_utils import tsutils


@tsutils.transform_args(statistic=tsutils.make_list)
@validate_arguments
@tsutils.doc(tsutils.docstrings)
def expanding_window(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    statistic: Optional[
        List[
            Literal[
                "corr",
                "count",
                "cov",
                "kurt",
                "max",
                "mean",
                "median",
                "min",
                "skew",
                "std",
                "sum",
                "var",
            ]
        ]
    ] = None,
    min_periods: PositiveInt = 1,
    center: bool = False,
    source_units=None,
    target_units=None,
    print_input=False,
):
    """Calculate an expanding window statistic.

    Parameters
    ----------
    statistic : str
        [optional, default is '']

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

    min_periods : int
        [optional, default is 1]

        Minimum number of observations in window required to have a value

    center : boolean
        [optional, default is False]

        Set the labels at the center of the window.

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

    ntsd = tsd.expanding(min_periods=min_periods, center=center)

    if statistic:
        nntsd = pd.DataFrame()
        for stat in statistic:
            ntsd = eval(f"ntsd.{stat}()")
            ntsd.columns = [
                tsutils.renamer(i, f"expanding.{stat}") for i in ntsd.columns
            ]
            nntsd = nntsd.join(ntsd, how="outer")
    else:
        nntsd = ntsd

    return tsutils.return_input(print_input, tsd, nntsd)
