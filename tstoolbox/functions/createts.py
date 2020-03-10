#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import pandas as pd

from .. import tsutils

warnings.filterwarnings("ignore")


@mando.command("createts", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def createts_cli(
    freq=None,
    fillvalue=None,
    input_ts=None,
    index_type="datetime",
    start_date=None,
    end_date=None,
    tablefmt="csv",
):
    """Create empty time series, optionally fill with a value.

    Parameters
    ----------
    freq : str
        [optional, default is None]

        To use this form `start_date` and `end_date` must be supplied
        also.  The `freq` option is the pandas date offset code used to create
        the index.

        Python example::

            freq='A'

        Command line example::

            --freq='A'

        {pandas_offset_codes}
    fillvalue
        [optional, default is None]

        The fill value for the time-series.  The default is None, which
        generates the date/time stamps only.
    {input_ts}
    {start_date}
    {end_date}
    {index_type}
    {tablefmt}

    """
    tsutils.printiso(
        createts(
            freq=freq,
            fillvalue=fillvalue,
            input_ts=input_ts,
            index_type=index_type,
            start_date=start_date,
            end_date=end_date,
        ),
        showindex="always",
        tablefmt=tablefmt,
    )


@tsutils.validator(freq=[str, ["pass", []], 1])
def createts(
    freq=None,
    fillvalue=None,
    input_ts=None,
    index_type="datetime",
    start_date=None,
    end_date=None,
):
    """Create empty time series, optionally fill with a value."""
    if input_ts is None:
        if (start_date is None) or (end_date is None) or (freq is None):
            raise ValueError(
                tsutils.error_wrapper(
                    """
If input_ts is None, then start_date, end_date, and freq must be supplied.

Instead you have:
start_date = {0},
end_date = {1},
freq = {2}
""".format(
                        start_date, end_date, freq
                    )
                )
            )

    if input_ts is not None:
        tsd = tsutils.common_kwds(
            tsutils.read_iso_ts(input_ts, index_type=index_type),
            start_date=start_date,
            end_date=end_date,
        )
        tsd = pd.DataFrame([fillvalue] * len(tsd.index), index=tsd.index)
    else:
        tindex = pd.date_range(start=start_date, end=end_date, freq=freq)
        tsd = pd.DataFrame([fillvalue] * len(tindex), index=tindex)
    return tsd


createts.__doc__ = createts_cli.__doc__
