#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import pandas as pd

from .. import tsutils


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def date_offset(years=0,
                months=0,
                weeks=0,
                days=0,
                hours=0,
                minutes=0,
                seconds=0,
                microseconds=0,
                columns=None,
                dropna='no',
                clean=False,
                skiprows=None,
                index_type='datetime',
                names=None,
                input_ts='-',
                start_date=None,
                end_date=None,
                source_units=None,
                target_units=None,
                round_index=None):
    """Apply an offset to a time-series.

    Parameters
    ----------
    years: number
        [optional, default is 0]

        Relative number of years to offset the datetime index,
        may be negative; adding or subtracting a relativedelta with
        relative information performs the corresponding arithmetic
        operation on the original datetime value with the information in
    months: number
        [optional, default is 0]

        Relative number of months to offset the datetime index,
        may be negative; adding or subtracting a relativedelta with
        relative information performs the corresponding arithmetic
        operation on the original datetime value with the information in
    weeks: number
        [optional, default is 0]

        Relative number of weeks to offset the datetime index,
        may be negative; adding or subtracting a relativedelta with
        relative information performs the corresponding arithmetic
        operation on the original datetime value with the information in
    days: number
        [optional, default is 0]

        Relative number of days to offset the datetime index,
        may be negative; adding or subtracting a relativedelta with
        relative information performs the corresponding arithmetic
        operation on the original datetime value with the information in
    hours: number
        [optional, default is 0]

        Relative number of hours to offset the datetime index,
        may be negative; adding or subtracting a relativedelta with
        relative information performs the corresponding arithmetic
        operation on the original datetime value with the information in
    minutes: number
        [optional, default is 0]

        Relative number of minutes to offset the datetime index,
        may be negative; adding or subtracting a relativedelta with
        relative information performs the corresponding arithmetic
        operation on the original datetime value with the information in
    seconds: number
        [optional, default is 0]

        Relative number of seconds to offset the datetime index,
        may be negative; adding or subtracting a relativedelta with
        relative information performs the corresponding arithmetic
        operation on the original datetime value with the information in
    microseconds: number
        [optional, default is 0]

        Relative number of microseconds to offset the datetime index,
        may be negative; adding or subtracting a relativedelta with
        relative information performs the corresponding arithmetic
        operation on the original datetime value with the information in
        the relativedelta.
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

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts,
                                                  skiprows=skiprows,
                                                  names=names,
                                                  index_type=index_type),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna='no',
                              source_units=source_units,
                              target_units=target_units,
                              clean=clean)

    relativedelta = pd.tseries.offsets.relativedelta
    ntsd = pd.DataFrame(tsd.values,
                        index=[i +
                               relativedelta(years=years,
                                             months=months,
                                             days=days,
                                             hours=hours,
                                             minutes=minutes,
                                             seconds=seconds,
                                             microseconds=microseconds)
                               for i in tsd.index])
    ntsd.columns = tsd.columns

    return tsutils.printiso(ntsd, showindex='always')
