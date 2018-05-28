#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import sys
import warnings
from builtins import range
from builtins import str

import mando
from mando.rst_text_formatter import RSTHelpFormatter

import pandas as pd

from . import tsutils

from .functions.plot import plot
from .functions.createts import createts
from .functions.filter import filter
from .functions.read import read
from .functions.date_slice import date_slice
from .functions.describe import describe
from .functions.peak_detection import peak_detection
from .functions.convert import convert
from .functions.equation import equation
from .functions.pick import pick
from .functions.stdtozrxp import stdtozrxp
from .functions.tstopickle import tstopickle
from .functions.accumulate import accumulate
from .functions.ewm_window import ewm_window
from .functions.expanding_window import expanding_window
from .functions.rolling_window import rolling_window
from .functions.aggregate import aggregate
from .functions.replace import replace
from .functions.clip import clip
from .functions.add_trend import add_trend
from .functions.remove_trend import remove_trend
from .functions.calculate_fdc import calculate_fdc
from .functions.stack import stack
from .functions.unstack import unstack
from .functions.fill import fill
from .functions.gof import gof

warnings.filterwarnings('ignore')


@mando.command()
def about():
    """Display version number and system information."""
    tsutils.about(__name__)


def _dtw(ts_a, ts_b, d=lambda x, y: abs(x - y), window=10000):
    """Return the DTW similarity distance timeseries numpy arrays.

    Arguments
    ---------
    ts_a, ts_b : array of shape [n_samples, n_timepoints]
        Two arrays containing n_samples of timeseries data
        whose DTW distance between each sample of A and B
        will be compared

    d : DistanceMetric object (default = abs(x-y))
        the distance measure used for A_i - B_j in the
        DTW dynamic programming function

    Returns
    -------
    DTW distance between A and B

    """
    # Create cost matrix via broadcasting with large int
    ts_a, ts_b = pd.np.array(ts_a), pd.np.array(ts_b)
    M, N = len(ts_a), len(ts_b)
    cost = sys.maxsize * pd.np.ones((M, N))

    # Initialize the first row and column
    cost[0, 0] = d(ts_a[0], ts_b[0])
    for i in range(1, M):
        cost[i, 0] = cost[i - 1, 0] + d(ts_a[i], ts_b[0])

    for j in range(1, N):
        cost[0, j] = cost[0, j - 1] + d(ts_a[0], ts_b[j])

    # Populate rest of cost matrix within window
    for i in range(1, M):
        for j in range(max(1, i - window),
                       min(N, i + window)):
            choices = cost[i - 1, j - 1], cost[i, j - 1], cost[i - 1, j]
            cost[i, j] = min(choices) + d(ts_a[i], ts_b[j])

    # Return DTW distance given window
    return cost[-1, -1]


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def dtw(input_ts='-',
        columns=None,
        start_date=None,
        end_date=None,
        round_index=None,
        dropna='no',
        window=10000):
    """Dynamic Time Warping.

    Parameters
    ----------
    window : int
         [optional, default is 10000]

         Window length.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {round_index}
    {dropna}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna='no')

    process = {}
    for i in tsd.columns:
        for j in tsd.columns:
            if (i, j) not in process and (j, i) not in process and i != j:
                process[(i, j)] = _dtw(tsd[i], tsd[j], window=window)

    ntsd = pd.DataFrame(list(process.items()))
    ncols = ntsd.columns
    ncols = ['Variables'] + [str(i) + 'DTW_score' for i in ncols[1:]]
    ntsd.columns = ncols
    return tsutils.printiso(ntsd)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def pca(input_ts='-',
        columns=None,
        start_date=None,
        end_date=None,
        n_components=None,
        round_index=None):
    """Return the principal components analysis of the time series.

    Does not return a time-series.

    Parameters
    ----------
    n_components : int
        [optional, default is None]

        The number of groups to separate the time series into.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {round_index}

    """
    from sklearn.decomposition import PCA

    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              round_index=round_index,
                              pick=columns)

    pca = PCA(n_components)
    pca.fit(tsd.dropna(how='any'))
    print(pca.components_)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def normalization(input_ts='-',
                  columns=None,
                  start_date=None,
                  end_date=None,
                  dropna='no',
                  mode='minmax',
                  min_limit=0,
                  max_limit=1,
                  pct_rank_method='average',
                  print_input=False,
                  round_index=None,
                  float_format='%g'):
    """Return the normalization of the time series.

    Parameters
    ----------
    mode : str
        [optional, default is 'minmax']

        minmax
            min_limit +
            (X-Xmin)/(Xmax-Xmin)*(max_limit-min_limit)

        zscore
            X-mean(X)/stddev(X)

        pct_rank
            rank(X)*100/N
    min_limit : float
        [optional, defaults to 0]

        Defines the minimum limit of the minmax normalization.
    max_limit : float
        [optional, defaults to 1]

        Defines the maximum limit of the minmax normalization.
    pct_rank_method : str
        [optional, defaults to 'average']

        Defines how tied ranks are broken.  Can be 'average', 'min', 'max',
        'first', 'dense'.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {print_input}
    {float_format}
    {round_index}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)

    # Trying to save some memory
    if print_input:
        otsd = tsd.copy()
    else:
        otsd = pd.DataFrame()

    if mode == 'minmax':
        tsd = (min_limit +
               (tsd - tsd.min()) /
               (tsd.max() - tsd.min()) *
               (max_limit - min_limit))
    elif mode == 'zscore':
        tsd = (tsd - tsd.mean()) / tsd.std()
    elif mode == 'pct_rank':
        tsd = tsd.rank(method=pct_rank_method, pct=True)
    else:
        raise ValueError("""
*
*   The 'mode' options are 'minmax', 'zscore', or 'pct_rank', you gave me
*   {0}.
*
""".format(mode))

    tsd = tsutils.memory_optimize(tsd)
    return tsutils.print_input(print_input,
                               otsd,
                               tsd,
                               '_{0}'.format(mode),
                               float_format=float_format)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def converttz(fromtz,
              totz,
              input_ts='-',
              columns=None,
              start_date=None,
              end_date=None,
              round_index=None,
              dropna='no'):
    """Convert the time zone of the index.

    Parameters
    ----------
    fromtz : str
        The time zone of the original time-series.
    totz : str
        The time zone of the converted time-series.
    {input_ts}
    {start_date}
    {end_date}
    {columns}
    {dropna}
    {round_index}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)
    tsd = tsd.tz_localize(fromtz).tz_convert(totz)
    return tsutils.printiso(tsd,
                            showindex='always')


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def convert_index_to_julian(epoch='julian',
                            input_ts='-',
                            columns=None,
                            start_date=None,
                            end_date=None,
                            round_index=None,
                            dropna='no'):
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

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)
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


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def pct_change(input_ts='-',
               columns=None,
               start_date=None,
               end_date=None,
               dropna='no',
               periods=1,
               fill_method='pad',
               limit=None,
               freq=None,
               print_input=False,
               round_index=None,
               float_format='%g'):
    """Return the percent change between times.

    Parameters
    ----------
    periods : int
        [optional, default is 1]

        The number of intervals to calculate percent change across.
    fill_method : str
        [optional, defaults to 'pad']

        Fill method for NA.  Defaults to 'pad'.
    limit
        [optional, defaults to None]

        Is the minimum number of consecutive NA values where no more filling
        will be made.
    freq : str
        [optional, defaults to None]

        A pandas time offset string to represent the interval.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {print_input}
    {float_format}
    {round_index}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)

    # Trying to save some memory
    if print_input:
        otsd = tsd.copy()
    else:
        otsd = pd.DataFrame()

    return tsutils.print_input(print_input,
                               otsd,
                               tsd.pct_change(periods=periods,
                                              fill_method=fill_method,
                                              limit=limit,
                                              freq=freq),
                               '_pct_change',
                               float_format=float_format)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def rank(input_ts='-',
         columns=None,
         start_date=None,
         end_date=None,
         dropna='no',
         axis=0,
         method='average',
         numeric_only=None,
         na_option='keep',
         ascending=True,
         pct=False,
         print_input=False,
         float_format='%g',
         round_index=None):
    """Compute numerical data ranks (1 through n) along axis.

    Equal values are assigned a rank that is the average of the ranks of those
    values

    Parameters
    ----------
    axis
        [optional, default is 0]

        0 or 'index' for rows. 1 or 'columns' for columns.  Index to direct
        ranking.
    method : str
        [optional, default is 'average']

        +-----------------+--------------------------------+
        | method argument | Description                    |
        +=================+================================+
        | average         | average rank of group          |
        +-----------------+--------------------------------+
        | min             | lowest rank in group           |
        +-----------------+--------------------------------+
        | max             | highest rank in group          |
        +-----------------+--------------------------------+
        | first           | ranks assigned in order they   |
        |                 | appear in the array            |
        +-----------------+--------------------------------+
        | dense           | like 'min', but rank always    |
        |                 | increases by 1 between groups  |
        +-----------------+--------------------------------+

    numeric_only
        [optional, default is None]

        Include only float, int, boolean data. Valid only for DataFrame or
        Panel objects.
    na_option : str
        [optional, default is 'keep']

        +--------------------+--------------------------------+
        | na_option argument | Description                    |
        +====================+================================+
        | keep               | leave NA values where they are |
        +--------------------+--------------------------------+
        | top                | smallest rank if ascending     |
        +--------------------+--------------------------------+
        | bottom             | smallest rank if descending    |
        +--------------------+--------------------------------+

    ascending
        [optional, default is True]

        False ranks by high (1) to low (N)
    pct
        [optional, default is False]

        Computes percentage rank of data.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {print_input}
    {float_format}
    {round_index}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)

    # Trying to save some memory
    if print_input:
        otsd = tsd.copy()
    else:
        otsd = pd.DataFrame()

    return tsutils.print_input(print_input,
                               otsd,
                               tsd.rank(axis=axis,
                                        method=method,
                                        numeric_only=numeric_only,
                                        na_option=na_option,
                                        ascending=ascending,
                                        pct=pct),
                               '_rank',
                               float_format=float_format)


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
                input_ts='-',
                start_date=None,
                end_date=None,
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

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna='no')

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


def main():
    """Main function."""
    if not os.path.exists('debug_tstoolbox'):
        sys.tracebacklimit = 0
    mando.main()


if __name__ == '__main__':
    main()
