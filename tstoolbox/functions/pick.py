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
def pick(columns,
         input_ts='-',
         start_date=None,
         end_date=None,
         round_index=None,
         dropna='no',
         skiprows=None,
         index_type='datetime',
         names=None,
         source_units=None,
         target_units=None,
         clean=False):
    """Will pick a column or list of columns from input.

    Can use column names or column numbers.  If using numbers, column
    number 1 is the first data column.

    Parameters
    ----------
    {columns}
    {input_ts}
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

    """
    return tsutils.printiso(
        tsutils.common_kwds(tsutils.read_iso_ts(input_ts,
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
                            clean=clean))
