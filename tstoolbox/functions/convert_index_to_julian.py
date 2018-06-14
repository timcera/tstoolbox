#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from builtins import str

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import pandas as pd

from .. import tsutils


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def convert_index_to_julian(epoch='julian',
                            input_ts='-',
                            columns=None,
                            start_date=None,
                            end_date=None,
                            round_index=None,
                            dropna='no',
                            clean=False):
    """Convert date/time index to Julian dates from different epochs.

    Parameters
    ----------
    epoch : str
        [optional, defaults to 'julian']

        Can be one of, 'julian', 'reduced', 'modified',
        'truncated', 'dublin', 'cnes', 'ccsds', 'lop', 'lilian', 'rata_die',
        'mars_sol_date', or a date and time.

        If supplying a date and time, most formats are recognized, however
        the closer the format is to ISO 8601 the better.  Also should check and
        make sure date was parsed as expected.  If supplying only a date, the
        epoch starts at midnight the morning of that date.

        +-----------+-------------------+----------------+---------------+
        | epoch     | Epoch             | Calculation    | Notes         |
        +===========+===================+================+===============+
        | julian    | 4713-01-01:12 BCE | JD             |               |
        +-----------+-------------------+----------------+---------------+
        | reduced   | 1858-11-16:12     | JD -           | [ [1]_ ]      |
        |           |                   | 2400000        | [ [2]_ ]      |
        +-----------+-------------------+----------------+---------------+
        | modified  | 1858-11-17:00     | JD -           | SAO 1957      |
        |           |                   | 2400000.5      |               |
        +-----------+-------------------+----------------+---------------+
        | truncated | 1968-05-24:00     | floor (JD -    | NASA 1979     |
        |           |                   | 2440000.5)     |               |
        +-----------+-------------------+----------------+---------------+
        | dublin    | 1899-12-31:12     | JD -           | IAU 1955      |
        |           |                   | 2415020        |               |
        +-----------+-------------------+----------------+---------------+
        | cnes      | 1950-01-01:00     | JD -           | CNES          |
        |           |                   | 2433282.5      | [ [3]_ ]      |
        +-----------+-------------------+----------------+---------------+
        | ccsds     | 1958-01-01:00     | JD -           | CCSDS         |
        |           |                   | 2436204.5      | [ [3]_ ]      |
        +-----------+-------------------+----------------+---------------+
        | lop       | 1992-01-01:00     | JD -           | LOP           |
        |           |                   | 2448622.5      | [ [3]_ ]      |
        +-----------+-------------------+----------------+---------------+
        | lilian    | 1582-10-15[13]    | floor (JD -    | Count of days |
        |           |                   | 2299159.5)     | of the        |
        |           |                   |                | Gregorian     |
        |           |                   |                | calendar      |
        +-----------+-------------------+----------------+---------------+
        | rata_die  | 0001-01-01[13]    | floor (JD -    | Count of days |
        |           | proleptic         | 1721424.5)     | of the        |
        |           | Gregorian         |                | Common        |
        |           | calendar          |                | Era           |
        +-----------+-------------------+----------------+---------------+
        | mars_sol  | 1873-12-29:12     | (JD - 2405522) | Count of      |
        |           |                   | /1.02749       | Martian days  |
        +-----------+-------------------+----------------+---------------+

        .. [1] . Hopkins, Jeffrey L. (2013). Using Commercial Amateur
           Astronomical Spectrographs, p. 257, Springer Science & Business
           Media, ISBN 9783319014425

        .. [2] . Palle, Pere L., Esteban, Cesar. (2014). Asteroseismology, p.
           185, Cambridge University Press, ISBN 9781107470620

        .. [3] . Theveny, Pierre-Michel. (10 September 2001). "Date Format"
           The TPtime Handbook. Media Lab.

    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {round_index}
    {dropna}
    {clean}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna,
                              clean=clean)
    allowed = {'julian': lambda x: x,
               'reduced': lambda x: x - 2400000,
               'modified': lambda x: x - 2400000.5,
               'truncated': lambda x: pd.np.floor(x - 2440000.5),
               'dublin': lambda x: x - 2415020,
               'cnes': lambda x: x - 2433282.5,
               'ccsds': lambda x: x - 2436204.5,
               'lop': lambda x: x - 2448622.5,
               'lilian': lambda x: pd.np.floor(x - 2299159.5),
               'rata_die': lambda x: pd.np.floor(x - 1721424.5),
               'mars_sol': lambda x: (x - 2405522) / 1.02749}
    try:
        tsd.index = allowed[epoch](tsd.index.to_julian_date())
    except KeyError:
        tsd.index = (tsd.index.to_julian_date() -
                     pd.to_datetime(tsutils.parsedate(epoch)).to_julian_date())

    tsd.index.name = '{0}_date'.format(epoch)
    tsd.index = tsd.index.format(formatter=lambda x: str('{0:f}'.format(x)))

    return tsutils.printiso(tsd,
                            showindex='always')
