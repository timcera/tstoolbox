#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from .. import tsutils


@mando.command("convert_index", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def convert_index_cli(
    to,
    interval=None,
    epoch="julian",
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    round_index=None,
    dropna="no",
    clean=False,
    names=None,
    source_units=None,
    target_units=None,
    skiprows=None,
    tablefmt="csv",
):
    """Convert datetime to/from Julian dates from different epochs.

    Parameters
    ----------
    to: str
        One of 'number' or 'datetime'.  If 'number', the source time-series
        should have a datetime index to convert to a number.  If 'datetime',
        source data should be a number and the converted index will be
        datetime.
    interval
        [optional, defaults to None, transformation]

        The `interval` parameter defines the unit time.  One of the pandas
        offset codes.  The default of 'None' will set the unit time for all
        defined epochs to daily except 'unix' which will default to seconds.

        You can give any smaller unit time than daily for all defined epochs
        except 'unix' which requires an interval less than seconds.  For an
        epoch that begins with an arbitrary date, you can use any interval
        equal to or smaller than the frequency of the time-series.

        {pandas_offset_codes}
    epoch : str
        [optional, defaults to 'julian', transformation]

        Can be one of, 'julian', 'reduced', 'modified', 'truncated', 'dublin',
        'cnes', 'ccsds', 'lop', 'lilian', 'rata_die', 'mars_sol_date', 'unix',
        or a date and time.

        If supplying a date and time, most formats are recognized, however
        the closer the format is to ISO 8601 the better.  Also should check and
        make sure date was parsed as expected.  If supplying only a date, the
        epoch starts at midnight the morning of that date.

        The 'unix' epoch uses a default `interval` of seconds, and all other
        defined epochs use a default `interval` of 'daily'.

        +-----------+----------------+----------------+-------------+
        | epoch     | Epoch          | Calculation    | Notes       |
        +===========+================+================+=============+
        | julian    | 4713-01-01:12  | JD             |             |
        |           | BCE            |                |             |
        +-----------+----------------+----------------+-------------+
        | reduced   | 1858-11-16:12  | JD -           | [ [1]_ ]    |
        |           |                | 2400000        | [ [2]_ ]    |
        +-----------+----------------+----------------+-------------+
        | modified  | 1858-11-17:00  | JD -           | SAO 1957    |
        |           |                | 2400000.5      |             |
        +-----------+----------------+----------------+-------------+
        | truncated | 1968-05-24:00  | floor (JD -    | NASA 1979,  |
        |           |                | 2440000.5)     | integer     |
        +-----------+----------------+----------------+-------------+
        | dublin    | 1899-12-31:12  | JD -           | IAU 1955    |
        |           |                | 2415020        |             |
        +-----------+----------------+----------------+-------------+
        | cnes      | 1950-01-01:00  | JD -           | CNES        |
        |           |                | 2433282.5      | [ [3]_ ]    |
        +-----------+----------------+----------------+-------------+
        | ccsds     | 1958-01-01:00  | JD -           | CCSDS       |
        |           |                | 2436204.5      | [ [3]_ ]    |
        +-----------+----------------+----------------+-------------+
        | lop       | 1992-01-01:00  | JD -           | LOP         |
        |           |                | 2448622.5      | [ [3]_ ]    |
        +-----------+----------------+----------------+-------------+
        | lilian    | 1582-10-15[13] | floor (JD -    | Count of    |
        |           |                | 2299159.5)     | days of the |
        |           |                |                | Gregorian   |
        |           |                |                | calendar,   |
        |           |                |                | integer     |
        +-----------+----------------+----------------+-------------+
        | rata_die  | 0001-01-01[13] | floor (JD -    | Count of    |
        |           | proleptic      | 1721424.5)     | days of the |
        |           | Gregorian      |                | Common      |
        |           | calendar       |                | Era,        |
        |           |                |                | integer     |
        +-----------+----------------+----------------+-------------+
        | mars_sol  | 1873-12-29:12  | (JD - 2405522) | Count of    |
        |           |                | /1.02749       | Martian     |
        |           |                |                | days        |
        +-----------+----------------+----------------+-------------+
        | unix      | 1970-01-01     | JD - 2440587.5 | seconds     |
        |           | T00:00:00      |                |             |
        +-----------+----------------+----------------+-------------+

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
    {skiprows}
    {names}
    {source_units}
    {target_units}
    {tablefmt}

    """
    tsd = convert_index(
        to,
        interval=interval,
        epoch=epoch,
        input_ts=input_ts,
        columns=columns,
        start_date=start_date,
        end_date=end_date,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        names=names,
        source_units=source_units,
        target_units=target_units,
        skiprows=skiprows,
    )
    tsutils.printiso(tsd, tablefmt=tablefmt)


@tsutils.validator(
    to=[str, ["domain", ["datetime", "number"]], 1],
    epoch=[
        [
            str,
            [
                "domain",
                [
                    "julian",
                    "reduced",
                    "modified",
                    "truncated",
                    "dublin",
                    "cnes",
                    "ccsds",
                    "lop",
                    "lilian",
                    "rata_die",
                    "mars_sol",
                    "unix",
                ],
            ],
            1,
        ],
        [tsutils.parsedate, ["pass", []], 1],
    ],
)
def convert_index(
    to,
    interval=None,
    epoch="julian",
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    round_index=None,
    dropna="no",
    clean=False,
    names=None,
    source_units=None,
    target_units=None,
    skiprows=None,
):
    """Convert datetime to/from Julian dates from different epochs."""
    # Clip to start_date/end_date if possible.
    if to == "datetime":
        index_type = "number"
        nstart_date = None
        nend_date = None
        nround_index = None
    elif to == "number":
        index_type = "datetime"
        nstart_date = start_date
        nend_date = end_date
        nround_index = round_index

    tsd = tsutils.common_kwds(
        tsutils.read_iso_ts(
            input_ts, skiprows=skiprows, names=names, index_type=index_type
        ),
        start_date=nstart_date,
        end_date=nend_date,
        pick=columns,
        round_index=nround_index,
        dropna=dropna,
        source_units=source_units,
        target_units=target_units,
        clean=clean,
    )

    allowed = {
        "julian": lambda x: x,
        "reduced": lambda x: x - 2400000,
        "modified": lambda x: x - 2400000.5,
        "truncated": lambda x: np.floor(x - 2440000.5),
        "dublin": lambda x: x - 2415020,
        "cnes": lambda x: x - 2433282.5,
        "ccsds": lambda x: x - 2436204.5,
        "lop": lambda x: x - 2448622.5,
        "lilian": lambda x: np.floor(x - 2299159.5),
        "rata_die": lambda x: np.floor(x - 1721424.5),
        "mars_sol": lambda x: (x - 2405522) / 1.02749,
        "unix": lambda x: (x - 2440587.5),
    }

    dailies = [
        "julian",
        "reduced",
        "modified",
        "truncated",
        "dublin",
        "cnes",
        "ccsds",
        "lop",
        "lilian",
        "rata_die",
        "mars_sol",
    ]

    epoch_dates = {
        "julian": "julian",
        "reduced": "1858-11-16T12",
        "modified": "1858-11-17T00",
        "truncated": "1968-05-24T00",
        "dublin": "1899-12-31T12",
        "cnes": "1950-01-01T00",
        "ccsds": "1958-01-01T00",
        "lop": "1992-01-01T00",
        "lilian": "1582-10-15T00",
        "rata_die": "0001-01-01T00",
        "mars_sol": "1873-12-29T12",
        "unix": "1970-01-01T00",
    }

    if interval is None:
        interval = "D"
    else:
        words = interval.split("-")
        if len(words) == 2:
            warnings.warn(
                """
*
*   The epoch keyword "{0}" overrides the anchoring suffix "{1}".
*
""".format(
                    epoch, words[1]
                )
            )

            interval = words[0]

    if epoch == "unix" and interval not in ["S", "s"]:
        warnings.warn(
            """
*
*   Typically the unix epoch would has an interval of 'S' (seconds).
*   Instead you gave {0}.
*
""".format(
                interval
            )
        )

    if epoch in dailies and interval != "D":
        warnings.warn(
            """
*
*   Typically the {0} epoch would has an interval of 'D' (days).
*   Instead you gave {1}.
*
""".format(
                epoch, interval
            )
        )

    if to == "number":
        # Index must be datetime - let's make sure
        tsd.index = pd.to_datetime(tsd.index)

        frac = to_offset("D").nanos / to_offset(interval).nanos

        try:
            tsd.index = allowed[epoch](tsd.index.to_julian_date()) * frac
        except KeyError:
            epoch_date = tsutils.parsedate(epoch)
            tsd.index = (
                tsd.index.to_julian_date() - epoch_date.to_julian_date()
            ) * frac

        if any(tsd.index != tsd.index.astype("int64")) is False:
            tsd.index = tsd.index.astype("int64")

    elif to == "datetime":
        tsd.index = pd.to_datetime(
            tsd.index.values, origin=epoch_dates.setdefault(epoch, epoch), unit=interval
        )

    if names is None:
        tsd.index.name = "{0}_date".format(epoch)

    if to == "datetime":
        index_type = "number"
        nstart_date = start_date
        nend_date = end_date
        nround_index = round_index
    elif to == "number":
        index_type = "datetime"
        nstart_date = None
        nend_date = None
        nround_index = None
    tsd = tsutils.common_kwds(
        tsd, start_date=nstart_date, end_date=nend_date, round_index=nround_index
    )
    return tsd


convert_index.__doc__ = convert_index_cli.__doc__
