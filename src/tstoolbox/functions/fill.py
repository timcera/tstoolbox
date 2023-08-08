"""A collection of filling routines."""

from typing import List, Literal, Optional, Union

import numpy as np
import pandas as pd
from pydantic import Field
from toolbox_utils import tsutils
from typing_extensions import Annotated


def _validate_columns(ntsd, from_columns, to_columns):
    from_columns = tsutils.common_kwds(ntsd, pick=from_columns)
    to_columns = tsutils.common_kwds(ntsd, pick=to_columns)
    for to in to_columns:
        for fro in from_columns:
            if to == fro:
                raise ValueError(
                    tsutils.error_wrapper(
                        f"""
You can't have columns in both "from_columns", and "to_columns"
keywords.  Instead you have "{to}" in both.
"""
                    )
                )
    return from_columns, to_columns


@tsutils.transform_args(from_columns=tsutils.make_list, to_columns=tsutils.make_list)
@tsutils.doc(tsutils.docstrings)
def fill(
    input_ts="-",
    method: Union[
        Literal[
            "ffill",
            "bfill",
            "linear",
            "index",
            "values",
            "nearest",
            "zero",
            "slinear",
            "quadratic",
            "cubic",
            "spline",
            "polynomial",
            "barycentric",
            "mean",
            "median",
            "max",
            "min",
            "from",
            "time",
            "krogh",
            "piecewise_polynomial",
            "from_derivatives",
            "pchip",
            "akima",
        ],
        float,
    ] = "ffill",
    print_input=False,
    start_date=None,
    end_date=None,
    columns=None,
    clean=False,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    skiprows=None,
    from_columns: Optional[List[Union[int, str]]] = None,
    to_columns: Optional[List[Union[int, str]]] = None,
    limit: Annotated[int, Field(ge=0)] = None,
    order: Annotated[int, Field(ge=0)] = None,
    force_freq: str = None,
):
    """Fill missing values (NaN) with different methods.

    Missing values can occur because of NaN, or because the time series
    is sparse.

    Parameters
    ----------
    method : str
        [optional, default is 'ffill']

        String contained in single quotes or a number that
        defines the method to use for filling.

        +----------------------+----------------------------------------------+
        | method=              | fill missing values with...                  |
        +======================+==============================================+
        | ffill                | ...the last good value                       |
        +----------------------+----------------------------------------------+
        | bfill                | ...the next good value                       |
        +----------------------+----------------------------------------------+
        | 2.3                  | ...with this number                          |
        +----------------------+----------------------------------------------+
        | linear               | ...ignore index, values are equally spaced   |
        +----------------------+----------------------------------------------+
        | index                | ...linear interpolation with datetime index  |
        +----------------------+----------------------------------------------+
        | values               | ...linear interpolation with numerical index |
        +----------------------+----------------------------------------------+
        | nearest              | ...nearest good value                        |
        +----------------------+----------------------------------------------+
        | zero                 | ...zeroth order spline                       |
        +----------------------+----------------------------------------------+
        | slinear              | ...first order spline                        |
        +----------------------+----------------------------------------------+
        | quadratic            | ...second order spline                       |
        +----------------------+----------------------------------------------+
        | cubic                | ...third order spline                        |
        +----------------------+----------------------------------------------+
        | spline               | ...nth order spline                          |
        | order=n              |                                              |
        +----------------------+----------------------------------------------+
        | polynomial           | ...nth order polynomial                      |
        | order=n              |                                              |
        +----------------------+----------------------------------------------+
        | barycentric          | ...barycentric                               |
        +----------------------+----------------------------------------------+
        | mean                 | ...with mean                                 |
        +----------------------+----------------------------------------------+
        | median               | ...with median                               |
        +----------------------+----------------------------------------------+
        | max                  | ...with maximum                              |
        +----------------------+----------------------------------------------+
        | min                  | ...with minimum                              |
        +----------------------+----------------------------------------------+
        | from                 | ...with good values from other columns       |
        +----------------------+----------------------------------------------+
        | time                 | ...daily and higher resolution to interval   |
        +----------------------+----------------------------------------------+
        | krogh                | ...krogh algorithm                           |
        +----------------------+----------------------------------------------+
        | piecewise_polynomial | ...piecewise-polynomial algorithm            |
        | from_derivatives     |                                              |
        +----------------------+----------------------------------------------+
        | pchip                | ...pchip algorithm                           |
        +----------------------+----------------------------------------------+
        | akima                | ...akima algorithm                           |
        +----------------------+----------------------------------------------+

    ${print_input}

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

    from_columns : str or list
        [required if method='from', otherwise not used]

        List of column names/numbers from which good values will be
        taken to fill missing values in the `to_columns` keyword.

    to_columns : str or list
        [required if method='from', otherwise not used]

        List of column names/numbers that missing values will be
        replaced in from good values in the `from_columns` keyword.

    limit : int
        [default is None]

        Gaps of missing values greater than this number will not be filled.

    order : int
        [required if method is 'spline' or 'polynomial', otherwise not used,
        default is None]

        The order of the 'spline' or 'polynomial' fit for missing values.

    ${tablefmt}

    ${force_freq}
        ${pandas_offset_codes}
    """
    tsd = tsutils.common_kwds(
        input_ts,
        dropna="all",
        skiprows=skiprows,
        names=names,
        index_type=index_type,
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        source_units=source_units,
        target_units=target_units,
        force_freq=force_freq,
        clean=clean,
    ).astype(float)
    ntsd = tsd.copy() if print_input is True else tsd
    ntsd = tsutils.asbestfreq(ntsd)
    offset = ntsd.index[1] - ntsd.index[0]
    predf = pd.DataFrame(
        dict(list(zip(tsd.columns, tsd.mean().values))), index=[tsd.index[0] - offset]
    )
    postf = pd.DataFrame(
        dict(list(zip(tsd.columns, tsd.mean().values))), index=[tsd.index[-1] + offset]
    )
    ntsd = pd.concat([predf, ntsd, postf])
    if method in ("ffill", "bfill"):
        ntsd = ntsd.fillna(method=method, limit=limit)
    elif method in (
        "linear",
        "time",
        "index",
        "values",
        "nearest",
        "zero",
        "slinear",
        "quadratic",
        "cubic",
        "spline",
        "polynomial",
        "barycentric",
        "kroch",
        "piecewise_polynomial",
        "pchip",
        "akima",
        "from_derivatives",
    ):
        ntsd = ntsd.interpolate(method=method, limit=limit, order=order)
    elif method == "mean":
        ntsd = ntsd.fillna(ntsd.mean(), limit=limit)
    elif method == "median":
        ntsd = ntsd.fillna(ntsd.median(), limit=limit)
    elif method == "max":
        ntsd = ntsd.fillna(ntsd.max(), limit=limit)
    elif method == "min":
        ntsd = ntsd.fillna(ntsd.min(), limit=limit)
    elif method == "from":
        from_columns, to_columns = _validate_columns(ntsd, from_columns, to_columns)
        for to in to_columns:
            for fro in from_columns:
                mask = ntsd.loc[:, to].isna()
                if len(mask) == 0:
                    break
                ntsd.loc[mask, to] = ntsd.loc[mask, fro]
    else:
        ntsd = ntsd.fillna(value=float(method), limit=limit)
    ntsd = ntsd.iloc[1:-1]
    tsd.index.name = "Datetime"
    ntsd.index.name = "Datetime"
    ntsd = tsutils.memory_optimize(ntsd)
    return tsutils.return_input(print_input, tsd, ntsd, "fill")


# @cltoolbox.command(formatter_class=RSTHelpFormatter)
def fill_by_correlation(
    method="move2",
    maximum_lag=0,
    transform="log10",
    choose_best="dtw",
    print_input=False,
    input_ts="-",
):
    """Fill missing values (NaN) with different methods.

    Missing values can occur because of NaN, or because the time series
    is sparse.

    :param method: String contained in single quotes or a number that
        defines the method to use for filling.  'move2': maintenance of
        variance extension - 2
    :param -p, --print_input: If set to 'True' will include the input
        columns in the output table.  Default is 'False'.
    :param -i, --input_ts <str>: Filename with data in 'ISOdate,value'
        format or '-' for stdin.
    """
    tsd = tsutils.common_kwds(input_ts)
    ntsd = tsd.copy() if print_input is True else tsd
    ntsd = tsutils.asbestfreq(ntsd)

    if transform == "log10":
        ntsd = np.log10(ntsd)

    firstcol = pd.DataFrame(ntsd.iloc[:, 0])
    basets = pd.DataFrame(ntsd.iloc[:, 1:])
    if choose_best is True:
        firstcol = pd.DataFrame(ntsd.iloc[:, 0])
        allothers = pd.DataFrame(ntsd.iloc[:, 1:])
        collect = []
        for index in list(range(maximum_lag + 1)):
            shifty = allothers.shift(index)
            testdf = firstcol.join(shifty)
            lagres = testdf.dropna().corr().iloc[1:, 0]
            collect.append(np.abs(lagres.values))
        collect = np.array(collect)
        bestlag, bestts = np.unravel_index(collect.argmax(), collect.shape)
        basets = pd.DataFrame(ntsd.iloc[:, bestts + 1].shift(bestlag))

    single_source_ts = ["move1", "move2", "move3"]
    if method.lower() in single_source_ts and len(basets.columns) != 1:
        raise ValueError(
            tsutils.error_wrapper(
                f"""
For methods in {single_source_ts}
You can only have a single source column.  You can pass in onlu 2
time-series or use the flag 'choose_best' along with 'maximum_lag'.
Instead there are {len(basets.columns)} source time series.
"""
            )
        )

    if method == "move1":
        ntsd = firstcol.join(basets)
        dna = ntsd.dropna()
        means = np.mean(dna)
        stdevs = np.std(dna)
        print(means[1] + stdevs[1] / stdevs[0] * means[0])
        print(means, stdevs)
