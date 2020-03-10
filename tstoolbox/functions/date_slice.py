#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils

warnings.filterwarnings("ignore")


@mando.command("date_slice", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def date_slice_cli(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    skiprows=None,
    index_type="datetime",
    names=None,
    round_index=None,
    source_units=None,
    target_units=None,
    float_format="g",
    tablefmt="csv",
):
    """Print out data to the screen between start_date and end_date.

    This isn't really useful anymore because "start_date" and "end_date"
    are available in all sub-commands.

    Parameters
    ----------
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {clean}
    {skiprows}
    {index_type}
    {names}
    {float_format}
    {source_units}
    {target_units}
    {round_index}
    {tablefmt}

    """
    tsutils.printiso(
        date_slice(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            round_index=round_index,
            source_units=source_units,
            target_units=target_units,
        ),
        float_format=float_format,
        tablefmt=tablefmt,
    )


def date_slice(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    skiprows=None,
    index_type="datetime",
    names=None,
    round_index=None,
    source_units=None,
    target_units=None,
):
    """Print out data to the screen between start_date and end_date."""
    return tsutils.common_kwds(
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


date_slice.__doc__ = date_slice_cli.__doc__
