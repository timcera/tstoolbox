#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import pandas as pd

from .. import tsutils

warnings.filterwarnings('ignore')


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def read(filenames,
         force_freq=None,
         append='columns',
         columns=None,
         start_date=None,
         end_date=None,
         dropna='no',
         skiprows=None,
         index_type='datetime',
         names=None,
         clean=False,
         source_units=None,
         target_units=None,
         float_format='%g',
         round_index=None,
         how='outer'):
    """Collect time series from a list of pickle or csv files.

    Prints the read in time-series in the tstoolbox standard format.

    Parameters
    ----------
    filenames : str
        List of comma delimited filenames to read time series
        from.
    how : str
        [optional, default is 'outer']

        Use PANDAS concept on how to join the separate DataFrames read
        from each file.  If how='outer' represents the union of the
        time-series, 'inner' is the intersection.
    append : str
        [optional, default is 'columns']

        The type of appending to do.  For "combine" option matching column
        indices will append rows, matching row indices will append columns, and
        matching column/row indices use the value from the first dataset.  You
        can use "row" to force an append along either axis.
    force_freq
        [optional]

        Force this frequency for the files.  Typically you will only want to
        enforce a smaller interval where tstoolbox will insert missing values
        as needed.  WARNING: you may lose data if not careful with this option.
        In general, letting the algorithm determine the frequency should always
        work, but this option will override.  Use PANDAS offset codes.

        {pandas_offset_codes}

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
    {float_format}
    {round_index}

    """
    assert append in ['combine', 'rows', 'columns'], """
*
*   The "append" keyword must be "combine", "rows", or "columns".
*   You game me {0}.
*
""".format(append)

    if force_freq is not None:
        dropna = 'no'

    filenames = filenames.split(',')
    result = pd.DataFrame()
    result_list = []
    for i in filenames:
        tsd = tsutils.common_kwds(
                                  tsutils.read_iso_ts(i,
                                                      skiprows=skiprows,
                                                      names=names,
                                                      index_type=index_type),
                                  start_date=start_date,
                                  end_date=end_date,
                                  pick=columns,
                                  round_index=round_index,
                                  dropna=dropna,
                                  force_freq=force_freq,
                                  clean=clean,
                                  source_units=source_units,
                                  target_units=target_units)
        if append != 'combine':
            result_list.append(tsd)
        else:
            result = result.combine_first(tsd)

    if append != 'combine':
        result = pd.concat(result_list, axis=append)

    result.sort_index(inplace=True)

    return tsutils.printiso(result,
                            float_format=float_format)
