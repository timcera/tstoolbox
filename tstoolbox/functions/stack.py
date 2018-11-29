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
def stack(input_ts='-',
          columns=None,
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
    """Return the stack of the input table.

    The stack command takes the standard table and
    converts to a three column table.

    From::

      Datetime,TS1,TS2,TS3
      2000-01-01 00:00:00,1.2,1018.2,0.0032
      2000-01-02 00:00:00,1.8,1453.1,0.0002
      2000-01-03 00:00:00,1.9,1683.1,-0.0004

    To::

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

    Parameters
    ----------
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

    newtsd = pd.DataFrame(tsd.stack()).reset_index(1)
    newtsd.columns = ['Columns', 'Values']
    newtsd = newtsd.groupby('Columns').apply(
        lambda d: d.sort_values('Columns')).reset_index('Columns', drop=True)
    return tsutils.printiso(newtsd)
