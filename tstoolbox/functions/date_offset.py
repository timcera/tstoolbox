#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import mando
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils


@mando.command("date_offset", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def date_offset_cli(
    intervals,
    offset,
    columns=None,
    dropna="no",
    clean=False,
    skiprows=None,
    index_type="datetime",
    names=None,
    input_ts="-",
    start_date=None,
    end_date=None,
    source_units=None,
    target_units=None,
    round_index=None,
    tablefmt="csv",
):
    """Apply a date offset to a time-series index.

    If you want to adjust to a different time-zone then should use the
    "converttz" tstoolbox command.

    Parameters
    ----------
    intervals: int

        Number of intervals of `offset` to shift the time index.  A positive
        integer moves the index forward, negative moves it backwards.

    offset: str

        Any of the Pandas offset codes.  This is only the offset code
        and doesn't include a prefixed interval.

        {pandas_offset_codes}

    {input_ts}
    {start_date}
    {end_date}
    {columns}
    {round_index}
    {dropna}
    {clean}
    {skiprows}
    {index_type}
    {source_units}
    {target_units}
    {names}
    {tablefmt}

    """
    tsutils.printiso(
        date_offset(
            intervals,
            offset,
            columns=columns,
            dropna=dropna,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            input_ts=input_ts,
            start_date=start_date,
            end_date=end_date,
            source_units=source_units,
            target_units=target_units,
            round_index=round_index,
        ),
        showindex="always",
        tablefmt=tablefmt,
    )


@tsutils.validator(intervals=[int, ["pass", []], 1], offset=[str, ["pass", []], 1])
def date_offset(
    intervals,
    offset,
    columns=None,
    dropna="no",
    clean=False,
    skiprows=None,
    input_ts="-",
    start_date=None,
    end_date=None,
    names=None,
    index_type="datetime",
    source_units=None,
    target_units=None,
    round_index=None,
):
    """Apply an offset to a time-series."""
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

    return tsd.tshift(intervals, offset)


date_offset.__doc__ = date_offset_cli.__doc__
