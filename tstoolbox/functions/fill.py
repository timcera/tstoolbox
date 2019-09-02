#!/sjr/beodata/local/python_linux/bin/python
"""A collection of filling routines."""

from __future__ import absolute_import, print_function

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import pandas as pd

from .. import tsutils


def _validate_columns(ntsd, from_columns, to_columns):
    from_columns = tsutils.common_kwds(ntsd, pick=from_columns)
    to_columns = tsutils.common_kwds(ntsd, pick=to_columns)
    for to in to_columns:
        for fro in from_columns:
            if to == fro:
                raise ValueError(
                    """
*
*   You can't have columns in both "from_columns", and "to_columns"
*   keywords.  Instead you have "{to}" in both.
*
""".format(
                        **locals()
                    )
                )
    return from_columns, to_columns


@mando.command("fill", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def fill_cli(
    input_ts="-",
    method="ffill",
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
    from_columns=None,
    to_columns=None,
    tablefmt="csv",
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

        +-----------+----------------------------------------+
        | method=   | fill missing values with...            |
        +===========+========================================+
        | ffill     | ...the last good value                 |
        +-----------+----------------------------------------+
        | bfill     | ...the next good value                 |
        +-----------+----------------------------------------+
        | 2.3       | ...with this number                    |
        +-----------+----------------------------------------+
        | linear    | ...with linearly interpolated values   |
        +-----------+----------------------------------------+
        | nearest   | ...nearest good value                  |
        +-----------+----------------------------------------+
        | zero      | ...zeroth order spline                 |
        +-----------+----------------------------------------+
        | slinear   | ...first order spline                  |
        +-----------+----------------------------------------+
        | quadratic | ...second order spline                 |
        +-----------+----------------------------------------+
        | cubic     | ...third order spline                  |
        +-----------+----------------------------------------+
        | mean      | ...with mean                           |
        +-----------+----------------------------------------+
        | median    | ...with median                         |
        +-----------+----------------------------------------+
        | max       | ...with maximum                        |
        +-----------+----------------------------------------+
        | min       | ...with minimum                        |
        +-----------+----------------------------------------+
        | from      | ...with good values from other columns |
        +-----------+----------------------------------------+

    {print_input}
    {input_ts}
    {start_date}
    {end_date}
    {clean}
    {skiprows}
    {index_type}
    {names}
    {source_units}
    {target_units}
    {columns}
    from_columns : str or list
        [required if method='from', otherwise not used]

        List of column names/numbers from which good values will be
        taken to fill missing values in the `to_columns` keyword.
    to_columns : str or list
        [required if method='from', otherwise not used]

        List of column names/numbers that missing values will be
        replaced in from good values in the `from_columns` keyword.
    {tablefmt}

    """
    tsutils._printiso(
        fill(
            input_ts=input_ts,
            method=method,
            print_input=print_input,
            start_date=start_date,
            end_date=end_date,
            columns=columns,
            clean=clean,
            index_type=index_type,
            names=names,
            source_units=source_units,
            target_units=target_units,
            skiprows=skiprows,
            from_columns=from_columns,
            to_columns=to_columns,
        ),
        tablefmt=tablefmt,
    )


def fill(
    input_ts="-",
    method="ffill",
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
    from_columns=None,
    to_columns=None,
):
    """Fill missing values (NaN) with different methods."""
    tsd = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts,
            dropna="all",
            skiprows=skiprows,
            names=names,
            index_type=index_type,
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        source_units=source_units,
        target_units=target_units,
        clean=clean,
    )
    if print_input is True:
        ntsd = tsd.copy()
    else:
        ntsd = tsd
    ntsd = tsutils.asbestfreq(ntsd)
    offset = ntsd.index[1] - ntsd.index[0]
    predf = pd.DataFrame(
        dict(list(zip(tsd.columns, tsd.mean().values))), index=[tsd.index[0] - offset]
    )
    postf = pd.DataFrame(
        dict(list(zip(tsd.columns, tsd.mean().values))), index=[tsd.index[-1] + offset]
    )
    ntsd = pd.concat([predf, ntsd, postf])
    if method in ["ffill", "bfill"]:
        ntsd = ntsd.fillna(method=method)
    elif method == "linear":
        ntsd = ntsd.apply(pd.Series.interpolate, method="values")
    elif method in ["nearest", "zero", "slinear", "quadratic", "cubic"]:
        from scipy.interpolate import interp1d

        for c in ntsd.columns:
            df2 = ntsd[c].dropna()
            f = interp1d(df2.index.values.astype("d"), df2.values, kind=method)
            slices = pd.isnull(ntsd[c])
            ntsd[c][slices] = f(ntsd[c][slices].index.values.astype("d"))
    elif method == "mean":
        ntsd = ntsd.fillna(ntsd.mean())
    elif method == "median":
        ntsd = ntsd.fillna(ntsd.median())
    elif method == "max":
        ntsd = ntsd.fillna(ntsd.max())
    elif method == "min":
        ntsd = ntsd.fillna(ntsd.min())
    elif method == "from":
        from_columns, to_columns = _validate_columns(ntsd, from_columns, to_columns)
        for to in to_columns:
            for fro in from_columns:
                mask = ntsd.loc[:, to].isna()
                if len(mask) == 0:
                    break
                ntsd.loc[mask, to] = ntsd.loc[mask, fro]
    else:
        try:
            ntsd = ntsd.fillna(value=float(method))
        except ValueError:
            raise ValueError(
                """
*
*   The allowable values for 'method' are 'ffill', 'bfill', 'linear',
*   'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'mean', 'median',
*   'max', 'min' or a number.  Instead you have {0}.
*
""".format(
                    method
                )
            )
    ntsd = ntsd.iloc[1:-1]
    tsd.index.name = "Datetime"
    ntsd.index.name = "Datetime"
    return tsutils.return_input(print_input, tsd, ntsd, "fill")


# @mando.command(formatter_class=RSTHelpFormatter)
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
    tsd = tsutils.read_iso_ts(input_ts)
    if print_input is True:
        ntsd = tsd.copy()
    else:
        ntsd = tsd
    ntsd = tsutils.asbestfreq(ntsd)

    if transform == "log10":
        ntsd = pd.np.log10(ntsd)

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
            collect.append(pd.np.abs(lagres.values))
        collect = pd.np.array(collect)
        bestlag, bestts = pd.np.unravel_index(collect.argmax(), collect.shape)
        basets = pd.DataFrame(ntsd.iloc[:, bestts + 1].shift(bestlag))

    single_source_ts = ["move1", "move2", "move3"]
    if method.lower() in single_source_ts:
        if len(basets.columns) != 1:
            raise ValueError(
                """
*
*   For methods in {0}
*   You can only have a single source column.  You can pass in onlu 2
*   time-series or use the flag 'choose_best' along with 'maximum_lag'.
*   Instead there are {1} source time series.
*
""".format(
                    single_source_ts, len(basets.columns)
                )
            )

    if method == "move1":
        ntsd = firstcol.join(basets)
        dna = ntsd.dropna()
        means = pd.np.mean(dna)
        stdevs = pd.np.std(dna)
        print(means[1] + stdevs[1] / stdevs[0] * means[0])
        print(means, stdevs)


fill.__doc__ = fill_cli.__doc__
