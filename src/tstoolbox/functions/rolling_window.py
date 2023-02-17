"""Collection of functions for the manipulation of time series."""

import warnings
from typing import List, Literal, Optional

import pandas as pd
from pydantic import conint, validate_arguments
from toolbox_utils import tsutils

warnings.filterwarnings("ignore")


@tsutils.transform_args(window=tsutils.make_list)
@validate_arguments
@tsutils.doc(tsutils.docstrings)
def rolling_window(
    statistic: Literal[
        "corr",
        "count",
        "cov",
        "kurt",
        "max",
        "mean",
        "median",
        "min",
        "quantile",
        "skew",
        "std",
        "sum",
        "var",
    ],
    groupby=None,
    window: Optional[List[conint(ge=0)]] = None,
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    span=None,
    min_periods: Optional[conint(ge=0)] = None,
    center: bool = False,
    win_type: Optional[str] = None,
    on: Optional[str] = None,
    closed: Optional[Literal["right", "left", "both", "neither"]] = None,
    source_units=None,
    target_units=None,
    print_input=False,
):
    """Calculate a rolling window statistic.

    Parameters
    ----------
    statistic : str
        The statistic that will be applied to each
        window.

        +----------+--------------------+
        | corr     | correlation        |
        +----------+--------------------+
        | count    | count of numbers   |
        +----------+--------------------+
        | cov      | covariance         |
        +----------+--------------------+
        | kurt     | kurtosis           |
        +----------+--------------------+
        | max      | maximum            |
        +----------+--------------------+
        | mean     | mean               |
        +----------+--------------------+
        | median   | median             |
        +----------+--------------------+
        | min      | minimum            |
        +----------+--------------------+
        | quantile | quantile           |
        +----------+--------------------+
        | skew     | skew               |
        +----------+--------------------+
        | std      | standard deviation |
        +----------+--------------------+
        | sum      | sum                |
        +----------+--------------------+
        | var      | variance           |
        +----------+--------------------+

    ${groupby}

    window
        [optional, default = 2]

        Size of the moving window. This is the number of observations used for
        calculating the statistic. Each window will be a fixed size.

        If it is an offset then this will be the time period of each window. Each
        window will be a variable sized based on the observations included in
        the time-period. This is only valid for datetimelike indexes.

    min_periods : int
        [optional, default is None]

        Minimum number of observations in window required to have a value
        (otherwise result is NA). For a window that is specified by an offset,
        this will default to 1.

    center : boolean
        [optional, default is False]

        Set the labels at the center of the window.

    win_type : str
        [optional, default is None]

        Provide a window type.

        One of::

            boxcar
            triang
            blackman
            hamming
            bartlett
            parzen
            bohman
            blackmanharris
            nuttall
            barthann
            kaiser (needs beta)
            gaussian (needs std)
            general_gaussian (needs power, width)
            slepian (needs width)
            exponential (needs tau), center is set to None.

    on : str
        [optional, default is None]

        For a DataFrame, column on which to calculate the rolling window,
        rather than the index

    closed : str
        [optional, default is None]

        Make the interval closed on the 'right', 'left', 'both' or 'neither'
        endpoints. For offset-based windows, it defaults to 'right'. For fixed
        windows, defaults to 'both'. Remaining cases not implemented for fixed
        windows.

    span :
        [optional, default = 2]

        DEPRECATED: Changed to 'window' to be consistent with pandas.

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
        groupby=groupby,
    )

    if span is not None:
        window = [span]
    if window is None:
        window = [2]

    ntsd = pd.DataFrame()
    for win in window:
        statstr = ""
        if statistic in (
            "corr",
            "count",
            "cov",
            "kurt",
            "max",
            "mean",
            "median",
            "min",
            "quantile",
            "skew",
            "std",
            "sum",
            "var",
        ):
            statstr = f".{statistic}()"
        etsd = eval(
            f"""tsd.apply(lambda x:
                       x.rolling({win},
                                 min_periods={min_periods},
                                 center={center},
                                 win_type={win_type},
                                 on={on},
                                 closed={closed}){statstr}
                                 )"""
        )
        etsd.columns = [
            tsutils.renamer(i, f"rolling.{win}.{statistic}") for i in etsd.columns
        ]

        ntsd = ntsd.join(etsd, how="outer")

    return tsutils.return_input(print_input, tsd, ntsd, None)
