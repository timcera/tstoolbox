#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import pandas as pd

from .. import tsutils


@mando.command("rank", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def rank_cli(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    axis=0,
    method="average",
    numeric_only=None,
    na_option="keep",
    ascending=True,
    pct=False,
    print_input=False,
    float_format="g",
    source_units=None,
    target_units=None,
    round_index=None,
    tablefmt="csv",
):
    """Compute numerical data ranks (1 through n) along axis.

    Equal values are assigned a rank depending on `method`.

    Parameters
    ----------
    axis
        [optional, default is 0]

        0 or 'index' for rows. 1 or 'columns' for columns.  Index to
        direct ranking.
    method : str
        [optional, default is 'average']

        +-----------------+--------------------------------+
        | method argument | Description                    |
        +=================+================================+
        | average         | average rank of group          |
        +-----------------+--------------------------------+
        | min             | lowest rank in group           |
        +-----------------+--------------------------------+
        | max             | highest rank in group          |
        +-----------------+--------------------------------+
        | first           | ranks assigned in order they   |
        |                 | appear in the array            |
        +-----------------+--------------------------------+
        | dense           | like 'min', but rank always    |
        |                 | increases by 1 between groups  |
        +-----------------+--------------------------------+

    numeric_only
        [optional, default is None]

        Include only float, int, boolean data. Valid only for DataFrame
        or Panel objects.
    na_option : str
        [optional, default is 'keep']

        +--------------------+--------------------------------+
        | na_option argument | Description                    |
        +====================+================================+
        | keep               | leave NA values where they are |
        +--------------------+--------------------------------+
        | top                | smallest rank if ascending     |
        +--------------------+--------------------------------+
        | bottom             | smallest rank if descending    |
        +--------------------+--------------------------------+

    ascending
        [optional, default is True]

        False ranks by high (1) to low (N)
    pct
        [optional, default is False]

        Computes percentage rank of data.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {skiprows}
    {index_type}
    {names}
    {clean}
    {print_input}
    {float_format}
    {source_units}
    {target_units}
    {round_index}
    {tablefmt}

    """
    tsutils.printiso(
        rank(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            clean=clean,
            axis=axis,
            method=method,
            numeric_only=numeric_only,
            na_option=na_option,
            ascending=ascending,
            pct=pct,
            print_input=print_input,
            source_units=source_units,
            target_units=target_units,
            round_index=round_index,
        ),
        float_format=float_format,
        tablefmt=tablefmt,
    )


@tsutils.validator(
    method=[str, ["domain", ["average", "min", "max", "first", "dense"]], 1],
    na_option=[str, ["domain", ["keep", "top", "bottom"]], 1],
    ascending=[bool, ["domain", [True, False]], 1],
    pct=[bool, ["domain", [True, False]], 1],
)
def rank(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    axis=0,
    method="average",
    numeric_only=None,
    na_option="keep",
    ascending=True,
    pct=False,
    print_input=False,
    source_units=None,
    target_units=None,
    round_index=None,
):
    """Compute numerical data ranks (1 through n) along axis."""
    tsd = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts, skiprows=skiprows, names=names, index_type=index_type
        ),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        source_units=source_units,
        target_units=target_units,
        clean=clean,
    )

    # Trying to save some memory
    if print_input:
        otsd = tsd.copy()
    else:
        otsd = pd.DataFrame()

    return tsutils.return_input(
        print_input,
        otsd,
        tsd.rank(
            axis=axis,
            method=method,
            numeric_only=numeric_only,
            na_option=na_option,
            ascending=ascending,
            pct=pct,
        ),
        "rank",
    )


rank.__doc__ = rank_cli.__doc__
