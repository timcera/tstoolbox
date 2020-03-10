#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils

warnings.filterwarnings("ignore")


@mando.command("replace", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def replace_cli(
    from_values,
    to_values,
    round_index=None,
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    source_units=None,
    target_units=None,
    print_input=False,
    tablefmt="csv",
):
    """Return a time-series replacing values with others.

    Parameters
    ----------
    from_values
        All values in this comma separated list are replaced with the
        corresponding value in to_values.  Use the string 'None' to
        represent a missing value.  If using 'None' as a from_value it
        might be easier to use the "fill" subcommand instead.
    to_values
        All values in this comma separated list are the replacement
        values corresponding one-to-one to the items in from_values.
        Use the string 'None' to represent a missing value.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {skiprows}
    {index_type}
    {names}
    {clean}
    {round_index}
    {source_units}
    {target_units}
    {print_input}
    {tablefmt}

    """
    tsutils.printiso(
        replace(
            from_values,
            to_values,
            round_index=round_index,
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            clean=clean,
            source_units=source_units,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


@tsutils.validator(
    from_values=[float, ["pass", []], None], to_values=[float, ["pass", []], None]
)
def replace(
    from_values,
    to_values,
    round_index=None,
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    source_units=None,
    target_units=None,
    print_input=False,
):
    """Return a time-series replacing values with others."""
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

    nfrom_values = tsutils.make_list(from_values)

    nto_values = tsutils.make_list(to_values)

    ntsd = tsd.replace(nfrom_values, nto_values)

    return tsutils.return_input(print_input, tsd, ntsd, "replace")


replace.__doc__ = replace_cli.__doc__
