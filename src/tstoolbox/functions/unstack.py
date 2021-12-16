# -*- coding: utf-8 -*-
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings
from typing import Union

import mando
import typic
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils

warnings.filterwarnings("ignore")


@mando.command("unstack", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def unstack_cli(
    column_names,
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    round_index=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    clean=False,
    tablefmt="csv",
):
    """Return the unstack of the input table.

    The unstack command takes the stacked table and converts to a
    standard tstoolbox table.

    From::

      Datetime,Columns,Values
      2000-01-01,TS1,1.2
      2000-01-02,TS1,1.8
      2000-01-03,TS1,1.9
      2000-01-01,TS2,1018.2
      2000-01-02,TS2,1453.1
      2000-01-03,TS2,1683.1
      2000-01-01,TS3,0.0032
      2000-01-02,TS3,0.0002
      2000-01-03,TS3,-0.0004

    To::

      Datetime,TS1,TS2,TS3
      2000-01-01,1.2,1018.2,0.0032
      2000-01-02,1.8,1453.1,0.0002
      2000-01-03,1.9,1683.1,-0.0004

    Parameters
    ----------
    column_names
        The column in the table that holds the column names
        of the unstacked data.
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
    {round_index}
    {tablefmt}

    """
    tsutils.printiso(
        unstack(
            column_names,
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            round_index=round_index,
            dropna=dropna,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            source_units=source_units,
            target_units=target_units,
            clean=clean,
        ),
        tablefmt=tablefmt,
    )


@typic.al
def unstack(
    column_names: Union[int, str],
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    round_index=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    clean=False,
):
    """Return the unstack of the input table."""
    tsd = tsutils.common_kwds(
        input_ts,
        skiprows=skiprows,
        names=names,
        index_type=index_type,
        pick=columns,
        bestfreq=False,
        clean=False,
    )

    try:
        newtsd = tsd.pivot_table(
            index=tsd.index,
            values=tsd.columns.drop(column_names),
            columns=column_names,
            aggfunc="first",
        )
    except ValueError:
        raise ValueError(
            tsutils.error_wrapper("""
Duplicate index (time stamp and '{}') where found.
Found these duplicate indices:
{}
""".format(column_names, tsd.index.get_duplicates())))

    newtsd.index.name = "Datetime"

    newtsd.columns = [
        "_".join(tuple(map(str, col))).rstrip("_")
        for col in newtsd.columns.values
    ]

    # Remove weird characters from column names
    newtsd.rename(
        columns=lambda x: "".join([i for i in str(x) if i not in "'\" "]))

    newtsd = tsutils.common_kwds(
        newtsd,
        start_date=start_date,
        end_date=end_date,
        dropna=dropna,
        clean=False,
        source_units=source_units,
        target_units=target_units,
        round_index=round_index,
    )

    return newtsd


unstack.__doc__ = unstack_cli.__doc__
