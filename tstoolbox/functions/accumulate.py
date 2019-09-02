#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils

warnings.filterwarnings("ignore")


@mando.command("accumulate", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def accumulate_cli(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    statistic="sum",
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    print_input=False,
    tablefmt="csv",
):
    """Calculate accumulating statistics.

    Parameters
    ----------
    statistic : str
        [optional, default is 'sum', transformation]

        'sum', 'max', 'min', 'prod'
    {input_ts}
    {start_date}
    {end_date}
    {skiprows}
    {names}
    {columns}
    {dropna}
    {clean}
    {source_units}
    {target_units}
    {round_index}
    {index_type}
    {print_input}
    {tablefmt}

    """
    tsutils._printiso(
        accumulate(
            input_ts=input_ts,
            skiprows=skiprows,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            statistic=statistic,
            round_index=round_index,
            index_type=index_type,
            names=names,
            source_units=source_units,
            target_units=target_units,
            print_input=print_input,
        ),
        tablefmt=tablefmt,
    )


def accumulate(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    statistic="sum",
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    print_input=False,
):
    """Calculate accumulating statistics."""
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
    try:
        ntsd = eval("tsd.cum{0}()".format(statistic))
    except AttributeError:
        raise ValueError(
            """
*
*   Statistic {0} is not implemented.
*
""".format(
                statistic
            )
        )
    return tsutils.return_input(print_input, tsd, ntsd, statistic)


accumulate.__doc__ = accumulate_cli.__doc__
