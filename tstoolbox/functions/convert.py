#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

from .. import tsutils

warnings.filterwarnings('ignore')


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def convert(input_ts='-',
            columns=None,
            start_date=None,
            end_date=None,
            dropna='no',
            clean=False,
            skiprows=None,
            index_type='datetime',
            names=None,
            factor=1.0,
            offset=0.0,
            print_input=False,
            round_index=None,
            source_units=None,
            target_units=None,
            float_format='%g'):
    """Convert values of a time series by applying a factor and offset.

    See the 'equation' subcommand for a generalized form of this
    command.

    Parameters
    ----------
    factor : float
        [optional, default is 1.0]

        Factor to multiply the time series values.
    offset : float
        [optional, default is 0.0]

        Offset to add to the time series values.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {clean}
    {skiprows}
    {index_type}
    {names}
    {print_input}
    {float_format}
    {source_units}
    {target_units}
    {round_index}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts,
                                                  skiprows=skiprows,
                                                  names=names,
                                                  index_type=index_type),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna,
                              source_units=source_units,
                              target_units=target_units,
                              clean=clean)
    tmptsd = tsd * factor + offset
    return tsutils.print_input(print_input, tsd, tmptsd, '_convert',
                               float_format=float_format)
