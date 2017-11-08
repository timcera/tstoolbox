#!/usr/bin/env python
"""Collection of functions for the manipulation of time series."""

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import sys
import os.path
import warnings

# The numpy import is needed like this to be able to include numpy
# functions in the 'equation' subcommand.
from numpy import *

import pandas as pd

import mando
from mando.rst_text_formatter import RSTHelpFormatter

from . import tsutils
from . import fill_functions

warnings.filterwarnings('ignore')
fill = fill_functions.fill

_offset_aliases = {
    86400000000000: 'D',
    604800000000000: 'W',
    2419200000000000: 'M',
    2505600000000000: 'M',
    2592000000000000: 'M',
    2678400000000000: 'M',
    31536000000000000: 'A',
    31622400000000000: 'A',
    3600000000000: 'H',
    60000000000: 'M',
    1000000000: 'T',
    1000000: 'L',
    1000: 'U',
}


@mando.command()
def about():
    """Display version number and system information.
    """
    tsutils.about(__name__)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def createts(freq=None,
             fillvalue=None,
             input_ts=None,
             start_date=None,
             end_date=None):
    """Create empty time series, optionally fill with a value.

    Parameters
    ----------
    freq
        To use this form `--start_date` and `--end_date` must be supplied
        also.  The `freq` option is the pandas date offset code used to create
        the index.
    fillvalue
        The fill value for the time-series.  The default is None, which
        generates the date/time stamps only.
    {input_ts}
    {start_date}
    {end_date}

    """
    if input_ts is not None:
        tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                                  start_date=start_date,
                                  end_date=end_date)
        tsd = pd.DataFrame([fillvalue] * len(tsd.index),
                           index=tsd.index)
    elif start_date is None or end_date is None or freq is None:
        raise ValueError("""
*
*   If input_ts is None, then start_date, end_date, and freq must be supplied.
*   Instead you have:
*   start_date = {0},
*   end_date = {1},
*   freq = {2}
*
""".format(start_date, end_date, freq))
    else:
        tindex = pd.date_range(start=start_date,
                               end=end_date,
                               freq=freq)
        tsd = pd.DataFrame([fillvalue] * len(tindex),
                           index=tindex)
    return tsutils.printiso(tsd,
                            showindex="always")


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def filter(filter_type,
           input_ts='-',
           columns=None,
           start_date=None,
           end_date=None,
           dropna='no',
           print_input=False,
           cutoff_period=None,
           window_len=5,
           float_format='%g',
           round_index=None):
    """Apply different filters to the time-series.

    Parameters
    ----------
    filter_type : str
        'flat', 'hanning', 'hamming', 'bartlett', 'blackman',
        'fft_highpass' and 'fft_lowpass' for Fast Fourier Transform
        filter in the frequency domain.
    window_len : int
        For the windowed types, 'flat', 'hanning', 'hamming',
        'bartlett', and 'blackman' specifies the length of the window.
        Defaults to 5.
    cutoff_period
        For 'fft_highpass' and 'fft_lowpass'. Default is None, but
        must be supplied if using 'fft_highpass' or 'fft_lowpass'.
        The period in input time units that will form the cutoff between
        low frequencies (longer periods) and high frequencies (shorter
        periods).  Filter will be smoothed by `window_len` running
        average.
    {input_ts}
    {start_date}
    {end_date}
    {columns}
    {float_format}
    {dropna}
    {round_index}
    {print_input}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)
    from tstoolbox import filters

    if len(tsd.values) < window_len:
        raise ValueError("""
*
*   Input vector (length={0}) needs to be bigger than window size ({1}).
*
""".format(len(tsd.values), window_len))

    # Trying to save some memory
    if print_input:
        otsd = tsd.copy()
    else:
        otsd = pd.DataFrame()

    for col in tsd.columns:
        # fft_lowpass, fft_highpass
        if filter_type == 'fft_lowpass':
            tsd[col].values[:] = filters._transform(tsd[col].values,
                                                    cutoff_period,
                                                    window_len,
                                                    lopass=True)
        elif filter_type == 'fft_highpass':
            tsd[col].values[:] = filters._transform(tsd[col].values,
                                                    cutoff_period,
                                                    window_len)
        elif filter_type in ['flat',
                             'hanning',
                             'hamming',
                             'bartlett',
                             'blackman']:
            if window_len < 3:
                continue
            s = pd.np.pad(tsd[col].values, window_len // 2, 'reflect')

            if filter_type == 'flat':  # moving average
                w = pd.np.ones(window_len, 'd')
            else:
                w = eval('pd.np.' + filter_type + '(window_len)')
            tsd[col].values[:] = pd.np.convolve(w / w.sum(), s, mode='valid')
        else:
            raise ValueError("""
*
*   Filter type {0} not implemented.
*
""".format(filter_type))
    return tsutils.print_input(print_input, otsd, tsd, '_filter',
                               float_format=float_format)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def read(filenames,
         columns=None,
         start_date=None,
         end_date=None,
         dropna='no',
         float_format='%g',
         round_index=None,
         how='outer'):
    """Collect time series from a list of pickle or csv files.

    Prints the read in time-series in the tstoolbox standard format.

    Parameters
    ----------
    filenames : str
        List of comma delimited filenames to read time series from.
    how : str
        Use PANDAS concept on how to join the separate DataFrames read
        from each file.  Default is how='outer' which is the union, 'inner'
        is the intersection.
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {float_format}
    {round_index}

    """
    filenames = filenames.split(',')
    result = pd.concat([tsutils.common_kwds(
        tsutils.read_iso_ts(i, extended_columns=True),
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna) for i in filenames], join=how, axis=1)

    colnames = ['.'.join(i.split('.')[1:]) for i in result.columns]
    if len(colnames) == len(set(colnames)):
        result.columns = colnames
    else:
        result.columns = [i if result.columns.tolist().count(i) == 1
                          else i + str(index)
                          for index, i in enumerate(result.columns)]

    return tsutils.printiso(result,
                            float_format=float_format)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def date_slice(input_ts='-',
               columns=None,
               start_date=None,
               end_date=None,
               dropna='no',
               round_index=None,
               float_format='%g'):
    """Print out data to the screen between start_date and end_date.

    Parameters
    ----------
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {float_format}
    {round_index}

    """
    return tsutils.printiso(
        tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                            start_date=start_date,
                            end_date=end_date,
                            pick=columns,
                            round_index=round_index,
                            dropna=dropna), float_format=float_format)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def describe(input_ts='-',
             columns=None,
             start_date=None,
             end_date=None,
             dropna='no',
             transpose=False):
    """Print out statistics for the time-series.

    Parameters
    ----------
    transpose
        If 'transpose' option is used, will transpose describe output.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              dropna=dropna)
    if transpose is True:
        ntsd = tsd.describe().transpose()
    else:
        ntsd = tsd.describe()

    ntsd.index.name = 'Statistic'
    return tsutils.printiso(ntsd,
                            showindex="always")


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def peak_detection(input_ts='-',
                   columns=None,
                   start_date=None,
                   end_date=None,
                   dropna='no',
                   method='rel',
                   extrema='peak',
                   window=24,
                   pad_len=5,
                   points=9,
                   lock_frequency=False,
                   float_format='%g',
                   round_index=None,
                   print_input=''):
    r"""Peak and valley detection.

    Parameters
    ----------
    extrema : str
        'peak', 'valley', or 'both' to determine what should be
        returned.  Default is 'peak'.
    method : str
        'rel', 'minmax', 'zero_crossing', 'parabola', 'sine' methods are
        available.  The different algorithms have different strengths
        and weaknesses.  The 'rel' algorithm is the default.
    window : int
        There will not usually be multiple peaks within the window
        number of values.  The different methods use this variable in
        different ways.  For 'rel' the window keyword specifies how many
        points on each side to require a comparator(n,n+x) = True.  For
        'minmax' the window keyword is the distance to look ahead from
        a peak candidate to determine if it is the actual peak.

        '(sample / period) / f'

        where f might be a good choice between 1.25 and 4.

        For 'zero_crossing' the window keyword is the dimension of the
        smoothing window and should be an odd integer.
    pad_len : int
        Used with FFT to pad edges of time-series.
    points : int
        For 'parabola' and 'sine' methods. How many points around the
        peak should be used during curve fitting, must be odd.  The
        Default is 9.
    lock_frequency
        For 'sine' method only.  Specifies if the frequency argument of
        the model function should be locked to the value calculated from
        the raw peaks or if optimization process may tinker with it.
        (default: False)
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {float_format}
    {round_index}
    {print_input}

    """
    # Couldn't get fft method working correctly.  Left pieces in
    # in case want to figure it out in the future.

    if extrema not in ['peak', 'valley', 'both']:
        raise ValueError("""
*
*   The `extrema` argument must be one of 'peak',
*   'valley', or 'both'.  You supplied {0}.
*
""".format(extrema))

    if method not in ['rel', 'minmax', 'zero_crossing', 'parabola', 'sine']:
        raise ValueError("""
*
*   The `method` argument must be one of 'rel', 'minmax',
*   'zero_crossing', 'parabola', or 'sine'.  You supplied {0}.
*
""".format(method))

    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)

    window = int(window)
    kwds = {}
    if method == 'rel':
        from tstoolbox.peakdetect import _argrel as func
        window = window / 2
        if window == 0:
            window = 1
        kwds['window'] = int(window)
    elif method == 'minmax':
        from tstoolbox.peakdetect import _peakdetect as func
        window = int(window / 2)
        if window == 0:
            window = 1
        kwds['window'] = int(window)
    elif method == 'zero_crossing':
        from tstoolbox.peakdetect import _peakdetect_zero_crossing as func
        if not window % 2:
            window = window + 1
        kwds['window'] = int(window)
    elif method == 'parabola':
        from tstoolbox.peakdetect import _peakdetect_parabola as func
        kwds['points'] = int(points)
    elif method == 'sine':
        from tstoolbox.peakdetect import _peakdetect_sine as func
        kwds['points'] = int(points)
        kwds['lock_frequency'] = lock_frequency
    elif method == 'fft':  # currently would never be used.
        from tstoolbox.peakdetect import _peakdetect_fft as func
        kwds['pad_len'] = int(pad_len)

    if extrema == 'peak':
        tmptsd = tsd.rename(columns=lambda x: str(x) + '_peak', copy=True)
    if extrema == 'valley':
        tmptsd = tsd.rename(columns=lambda x: str(x) + '_valley', copy=True)
    if extrema == 'both':
        tmptsd = tsd.rename(columns=lambda x: str(x) + '_peak', copy=True)
        tmptsd = tmptsd.join(
            tsd.rename(columns=lambda x: str(x) + '_valley', copy=True),
            how='outer')

    for cols in tmptsd.columns:
        if method in ['fft', 'parabola', 'sine']:
            maxpeak, minpeak = func(
                tmptsd[cols].values, range(len(tmptsd[cols])), **kwds)
        else:
            maxpeak, minpeak = func(tmptsd[cols].values, **kwds)
        if cols[-5:] == '_peak':
            datavals = maxpeak
        if cols[-7:] == '_valley':
            datavals = minpeak
        maxx, _ = list(zip(*datavals))
        hold = tmptsd[cols][pd.np.array(maxx).astype('i')]
        tmptsd[cols][:] = pd.np.nan
        tmptsd[cols][pd.np.array(maxx).astype('i')] = hold

    tmptsd.index.name = 'Datetime'
    tsd.index.name = 'Datetime'
    return tsutils.print_input(print_input, tsd, tmptsd, None,
                               float_format=float_format)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def convert(input_ts='-',
            columns=None,
            start_date=None,
            end_date=None,
            dropna='no',
            factor=1.0,
            offset=0.0,
            print_input=False,
            round_index=None,
            float_format='%g'):
    """Convert values of a time series by applying a factor and offset.

    See the 'equation' subcommand for a generalized form of this
    command.

    Parameters
    ----------
    factor : float
        Factor to multiply the time series values.
    offset : float
        Offset to add to the time series values.
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
    tmptsd = tsd * factor + offset
    return tsutils.print_input(print_input, tsd, tmptsd, '_convert',
                               float_format=float_format)


def _parse_equation(equation_str):
    """Private function to parse the equation used in the calculations."""
    import re
    # Get rid of spaces
    nequation = equation_str.replace(' ', '')

    # Does the equation contain any x[t]?
    tsearch = re.search(r'\[.*?t.*?\]', nequation)

    # Does the equation contain any x1, x2, ...etc.?
    nsearch = re.search(r'x[1-9][0-9]*', nequation)

    # This beasty is so users can use 't' in their equations
    # Indices of 'x' are a function of 't' and can possibly be negative or
    # greater than the length of the DataFrame.
    # Correctly (I think) handles negative indices and indices greater
    # than the length by setting to nan
    # AssertionError happens when index negative.
    # IndexError happens when index is greater than the length of the
    # DataFrame.
    # UGLY!

    # testeval is just a list of the 't' expressions in the equation.
    # for example 'x[t]+0.6*max(x[t+1],x[t-1]' would have
    # testeval = ['t', 't+1', 't-1']
    testeval = set()
    # If there is both function of t and column terms x1, x2, ...etc.
    if tsearch and nsearch:
        testeval.update(re.findall(r'x[1-9][0-9]*\[(.*?t.*?)\]',
                                   nequation))
        # replace 'x1[t+1]' with 'x.iloc[t+1,1-1]'
        # replace 'x2[t+7]' with 'x.iloc[t+7,2-1]'
        # ...etc
        nequation = re.sub(r'x([1-9][0-9]*)\[(.*?t.*?)\]',
                           r'x.iloc[\2,\1-1]',
                           nequation)
        # replace 'x1' with 'x.iloc[t,1-1]'
        # replace 'x4' with 'x.iloc[t,4-1]'
        nequation = re.sub(r'x([1-9][0-9]*)',
                           r'x.iloc[t,\1-1]',
                           nequation)
    # If there is only a function of t, i.e. x[t]
    elif tsearch:
        testeval.update(re.findall(r'x\[(.*?t.*?)\]',
                                   nequation))
        nequation = re.sub(r'x\[(.*?t.*?)\]',
                           r'xxiloc[\1,:]',
                           nequation)
        # Replace 'x' with underlying equation, but not the 'x' in a word,
        # like 'maximum'.
        nequation = re.sub(r'(?<![a-zA-Z])x(?![a-zA-Z\[])',
                           r'xxiloc[t,:]',
                           nequation)
        nequation = re.sub(r'xxiloc',
                           r'x.iloc',
                           nequation)

    elif nsearch:
        nequation = re.sub(r'x([1-9][0-9]*)',
                           r'x.iloc[:,\1-1]',
                           nequation)

    try:
        testeval.remove('t')
    except KeyError:
        pass
    return tsearch, nsearch, testeval, nequation


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def equation(equation_str,
             input_ts='-',
             columns=None,
             start_date=None,
             end_date=None,
             dropna='no',
             print_input='',
             round_index=None,
             float_format='%g'):
    """Apply <equation_str> to the time series data.

    The <equation_str> argument is a string contained in single quotes
    with 'x' used as the variable representing the input.  For example,
    '(1 - x)*sin(x)'.

    Parameters
    ----------
    equation_str : str
        String contained in single quotes that defines the equation.

        There are four different types of equations that can be used.

        +-----------------------+-----------+-------------------------+
        | Description           | Variables | Examples                |
        +=======================+===========+=========================+
        | Equation applied to   | x         | x*0.3+4-x**2            |
        | all values in the     |           | sin(x)+pi*x             |
        | dataset.  Returns     |           |                         |
        | same number of        |           |                         |
        | columns as input.     |           |                         |
        +-----------------------+-----------+-------------------------+
        | Equation used time    | x and t   | 0.6*max(x[t-1],x[t+1])  |
        | relative to current   |           |                         |
        | record.  Applies      |           |                         |
        | equation to each      |           |                         |
        | column.  Returns same |           |                         |
        | number of columns as  |           |                         |
        | input.                |           |                         |
        +-----------------------+-----------+-------------------------+
        | Equation uses values  | x1, x2,   | x1+x2                   |
        | from different        | x3, ...   |                         |
        | columns.  Returns a   | xN        |                         |
        | single column.        |           |                         |
        +-----------------------+-----------+-------------------------+
        | Equation uses values  | x1, x2,   | x1[t-1]+x2+x3[t+1]      |
        | from different        | x3,       |                         |
        | columns and values    | ...xN, t  |                         |
        | from different rows.  |           |                         |
        | Returns a single      |           |                         |
        | column.               |           |                         |
        +-----------------------+-----------+-------------------------+

        Mathematical functions in the 'np' (numpy) name space can be
        used.  Additional examples::

            'x*4 + 2',
            'x**2 + cos(x)', and
            'tan(x*pi/180)'

        are all valid <equation> strings.  The variable 't' is special
        representing the time at which 'x' occurs.  This means you can
        do things like::

            'x[t] + max(x[t-1], x[t+1])*0.6'

        to add to the current value 0.6 times the maximum adjacent
        value.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {print_input}
    {float_format}
    {round_index}

    """
    x = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                            start_date=start_date,
                            end_date=end_date,
                            pick=columns,
                            round_index=round_index,
                            dropna=dropna)

    def returnval(t, x, testeval, nequation):
        for tst in testeval:
            tvalue = eval(tst)
            if tvalue < 0 or tvalue >= len(x):
                return pd.np.nan
        return eval(nequation)

    tsearch, nsearch, testeval, nequation = _parse_equation(equation_str)
    if tsearch and nsearch:
        y = pd.DataFrame(pd.Series(index=x.index),
                         columns=['_'],
                         dtype='float64')
        for t in range(len(x)):
            y.iloc[t, 0] = returnval(t, x, testeval, nequation)
    elif tsearch:
        y = x.copy()
        for t in range(len(x)):
            y.iloc[t, :] = returnval(t, x, testeval, nequation)
    elif nsearch:
        y = pd.DataFrame(pd.Series(index=x.index),
                         columns=['_'],
                         dtype='float64')
        y.iloc[:, 0] = eval(nequation)
    else:
        y = eval(equation_str)

    y = tsutils.memory_optimize(y)

    return tsutils.print_input(print_input,
                               x,
                               y,
                               '_equation',
                               float_format=float_format)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def pick(columns,
         input_ts='-',
         start_date=None,
         end_date=None,
         round_index=None,
         dropna='no'):
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
    {round_index}

    """
    return tsutils.printiso(
        tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                            start_date=start_date,
                            end_date=end_date,
                            pick=columns,
                            round_index=round_index,
                            dropna=dropna))


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def stdtozrxp(input_ts='-',
              columns=None,
              start_date=None,
              end_date=None,
              dropna='no',
              round_index=None,
              rexchange=None):
    """Print out data to the screen in a WISKI ZRXP format.

    Parameters
    ----------
    rexchange
        The REXCHANGE ID to be written into the zrxp header.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {round_index}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)
    if len(tsd.columns) > 1:
        raise ValueError("""
*
*   The "stdtozrxp" command can only accept a single
*   'time-series, instead it is seeing {0}.
*
""".format(len(tsd.columns)))
    if rexchange:
        print('#REXCHANGE{0}|*|'.format(rexchange))
    for i in range(len(tsd)):
        print(('{0.year:04d}{0.month:02d}{0.day:02d}{0.hour:02d}'
               '{0.minute:02d}{0.second:02d}, {1}').format(
                   tsd.index[i], tsd[tsd.columns[0]][i]))


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def tstopickle(filename,
               input_ts='-',
               columns=None,
               start_date=None,
               end_date=None,
               round_index=None,
               dropna='no'):
    """Pickle the data into a Python pickled file.

    Can be brought back into Python with 'pickle.load' or 'numpy.load'.
    See also 'tstoolbox read'.

    Parameters
    ----------
    filename : str
         The filename to store the pickled data.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {round_index}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)
    tsd.to_pickle(filename)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def accumulate(input_ts='-',
               columns=None,
               start_date=None,
               end_date=None,
               dropna='no',
               statistic='sum',
               round_index=None,
               print_input=False):
    """Calculate accumulating statistics.

    Parameters
    ----------
    statistic : str
        'sum', 'max', 'min', 'prod', defaults to 'sum'.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {print_input}
    {round_index}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)
    try:
        ntsd = eval('tsd.cum{0}()'.format(statistic))
    except AttributeError:
        raise ValueError("""
*
*   Statistic {0} is not implemented.
*
""".format(statistic))
    return tsutils.print_input(print_input, tsd, ntsd, '_' + statistic)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def rolling_window(input_ts='-',
                   columns=None,
                   start_date=None,
                   end_date=None,
                   dropna='no',
                   span=None,
                   statistic='mean',
                   wintype=None,
                   center=False,
                   print_input=False,
                   freq=None,
                   groupby=None):
    """Calculate a rolling window statistic.

    Parameters
    ----------
    span
        The number of previous intervals to include in the calculation
        of the statistic. If `span` is equal to 0 will give an expanding
        rolling window.  Defaults to 2.
    statistic : str
        For rolling window (span>0) and expanding window (span==0), one
        of 'count', 'sum', 'mean', 'median', 'min', 'max', 'std', 'var',
        'skew', 'kurt', 'quantile'.  For exponentially weighted windows
        have 'ewma' for mean average, 'ewvar' for variance, and 'ewmstd'
        for standard deviation
    wintype : str
        The 'mean' and 'sum' `statistic` calculation can also be
        weighted according to the `wintype` windows.  Some of the
        following windows require additional keywords identified in
        parenthesis: 'boxcar', 'triang', 'blackman', 'hamming',
        'bartlett', 'parzen', 'bohman', 'blackmanharris', 'nuttall',
        'barthann', 'kaiser' (needs beta), 'gaussian' (needs std),
        'general_gaussian' (needs power, width) 'slepian' (needs width).
    center
        If set to 'True' the calculation will be made for the value at
        the center of the window.  Default is 'False'.
    groupby
        Time offset to groupby.  Any PANDAS time offset.  This option
        supports what is probably an unusual situation where the
        rolling_window is performed separately within each groupby
        period.

        +-------+-----------------------------+
        | Alias | Description                 |
        +=======+=============================+
        | B     | business day                |
        +-------+-----------------------------+
        | C     | custom business day         |
        |       | (experimental)              |
        +-------+-----------------------------+
        | D     | calendar day                |
        +-------+-----------------------------+
        | W     | weekly                      |
        +-------+-----------------------------+
        | M     | month end                   |
        +-------+-----------------------------+
        | BM    | business month end          |
        +-------+-----------------------------+
        | CBM   | custom business month end   |
        +-------+-----------------------------+
        | MS    | month start                 |
        +-------+-----------------------------+
        | BMS   | business month start        |
        +-------+-----------------------------+
        | CBMS  | custom business month start |
        +-------+-----------------------------+
        | Q     | quarter end                 |
        +-------+-----------------------------+
        | BQ    | business quarter end        |
        +-------+-----------------------------+
        | QS    | quarter start               |
        +-------+-----------------------------+
        | BQS   | business quarter start      |
        +-------+-----------------------------+
        | A     | year end                    |
        +-------+-----------------------------+
        | BA    | business year end           |
        +-------+-----------------------------+
        | AS    | year start                  |
        +-------+-----------------------------+
        | BAS   | business year start         |
        +-------+-----------------------------+
        | H     | hourly                      |
        +-------+-----------------------------+
        | T     | minutely                    |
        +-------+-----------------------------+
        | S     | secondly                    |
        +-------+-----------------------------+
        | L     | milliseconds                |
        +-------+-----------------------------+
        | U     | microseconds                |
        +-------+-----------------------------+
        | N     | nanoseconds                 |
        +-------+-----------------------------+

        Weekly has the following anchored frequencies:

        +-------+-------------------------------+
        | Alias | Description                   |
        +=======+===============================+
        | W-SUN | weekly frequency (sundays).   |
        |       | Same as 'W'.                  |
        +-------+-------------------------------+
        | W-MON | weekly frequency (mondays)    |
        +-------+-------------------------------+
        | W-TUE | weekly frequency (tuesdays)   |
        +-------+-------------------------------+
        | W-WED | weekly frequency (wednesdays) |
        +-------+-------------------------------+
        | W-THU | weekly frequency (thursdays)  |
        +-------+-------------------------------+
        | W-FRI | weekly frequency (fridays)    |
        +-------+-------------------------------+
        | W-SAT | weekly frequency (saturdays)  |
        +-------+-------------------------------+

        Quarterly frequencies (Q, BQ, QS, BQS) and annual frequencies
        (A, BA, AS, BAS) have the following anchoring suffixes:

        +-------+-------------------------------+
        | Alias | Description                   |
        +=======+===============================+
        | -DEC  | year ends in December (same   |
        |       | as 'Q' and 'A')               |
        +-------+-------------------------------+
        | -JAN  | year ends in January          |
        +-------+-------------------------------+
        | -FEB  | year ends in February         |
        +-------+-------------------------------+
        | -MAR  | year ends in March            |
        +-------+-------------------------------+
        | -APR  | year ends in April            |
        +-------+-------------------------------+
        | -MAY  | year ends in May              |
        +-------+-------------------------------+
        | -JUN  | year ends in June             |
        +-------+-------------------------------+
        | -JUL  | year ends in July             |
        +-------+-------------------------------+
        | -AUG  | year ends in August           |
        +-------+-------------------------------+
        | -SEP  | year ends in September        |
        +-------+-------------------------------+
        | -OCT  | year ends in October          |
        +-------+-------------------------------+
        | -NOV  | year ends in November         |
        +-------+-------------------------------+

        Defaults to None.
    freq
        string or DateOffset object, optional (default None) Frequency
        to conform the data to before computing the statistic. Specified
        as a frequency string or DateOffset object.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {print_input}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              groupby=groupby,
                              dropna=dropna)

    if span is None:
        span = 2

    def _process_tsd(tsd,
                     statistic='mean',
                     span=None,
                     center=False,
                     wintype=None,
                     freq=None):
        try:
            span = int(span)
        except ValueError:
            span = len(tsd)
        window_list = [
            'boxcar',
            'triang',
            'blackman',
            'hamming',
            'bartlett',
            'parzen',
            'bohman',
            'blackmanharris',
            'nuttall',
            'barthann',
            'kaiser',
            'gaussian',
            'general_gaussian',
            'slepian',
        ]
        try:
            if wintype in window_list and statistic in ['mean', 'sum']:
                meantest = statistic == 'mean'
                newts = pd.stats.moments.rolling_window(tsd,
                                                        span,
                                                        wintype,
                                                        center=center,
                                                        mean=meantest,
                                                        freq=freq)
            elif statistic[:3] == "ewm":
                newts = eval('pd.stats.moments.{0}'
                             '(tsd, span=span, center=center, freq=freq)'
                             ''.format(statistic))
            else:
                if span == 0:
                    newts = eval('pd.stats.moments.expanding_{0}'
                                 '(tsd, center=center, freq=freq)'
                                 ''.format(statistic))
                else:
                    newts = eval('pd.stats.moments.rolling_{0}'
                                 '(tsd, span, center=center, freq=freq)'
                                 ''.format(statistic))
        except AttributeError:
            raise ValueError("""
*
*   Statistic '{0}' is not implemented.
*
""".format(statistic))
        return newts

    tmptsd = []
    if isinstance(tsd, pd.DataFrame):
        for nspan in str(span).split(','):
            tmptsd.append(_process_tsd(tsd,
                                       statistic=statistic,
                                       span=nspan,
                                       center=center,
                                       wintype=wintype,
                                       freq=freq))
    else:
        for nspan in str(span).split(','):
            jtsd = pd.DataFrame()
            for _, gb in tsd:
                # Assume span should be yearly if 365 or 366 and you have
                # groupby yearly.
                xspan = nspan
                if len(gb) in [365, 366] and int(nspan) in [365, 366]:
                    xspan = len(gb)
                jtsd = jtsd.append(_process_tsd(gb,
                                                statistic=statistic,
                                                span=xspan,
                                                center=center,
                                                wintype=wintype,
                                                freq=freq))
            tmptsd.append(jtsd)
    ntsd = pd.concat(tmptsd, join='outer', axis=1)

    ntsd.columns = [i[0] + '_' + i[1] for i in zip(ntsd.columns,
                                                   str(span).split(','))]
    return tsutils.print_input(print_input,
                               tsd,
                               ntsd,
                               '_' + statistic)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def aggregate(input_ts='-',
              columns=None,
              start_date=None,
              end_date=None,
              dropna='no',
              statistic='mean',
              agg_interval='daily',
              ninterval=1,
              round_index=None,
              print_input=False):
    """Take a time series and aggregate to specified frequency.

    Parameters
    ----------
    statistic : str
        'mean', 'sum', 'std', 'max', 'min', 'median', 'first', or 'last'
        to calculate the aggregation, defaults to 'mean'.  Can also be
        a comma separated list of statistic methods.
    agg_interval : str
        The interval to aggregate the time series.  Any of the PANDAS
        offset codes.

        +-------+-----------------------------+
        | Alias | Description                 |
        +=======+=============================+
        | B     | business day                |
        +-------+-----------------------------+
        | C     | custom business day         |
        |       | (experimental)              |
        +-------+-----------------------------+
        | D     | calendar day                |
        +-------+-----------------------------+
        | W     | weekly                      |
        +-------+-----------------------------+
        | M     | month end                   |
        +-------+-----------------------------+
        | BM    | business month end          |
        +-------+-----------------------------+
        | CBM   | custom business month end   |
        +-------+-----------------------------+
        | MS    | month start                 |
        +-------+-----------------------------+
        | BMS   | business month start        |
        +-------+-----------------------------+
        | CBMS  | custom business month start |
        +-------+-----------------------------+
        | Q     | quarter end                 |
        +-------+-----------------------------+
        | BQ    | business quarter end        |
        +-------+-----------------------------+
        | QS    | quarter start               |
        +-------+-----------------------------+
        | BQS   | business quarter start      |
        +-------+-----------------------------+
        | A     | year end                    |
        +-------+-----------------------------+
        | BA    | business year end           |
        +-------+-----------------------------+
        | AS    | year start                  |
        +-------+-----------------------------+
        | BAS   | business year start         |
        +-------+-----------------------------+
        | H     | hourly                      |
        +-------+-----------------------------+
        | T     | minutely                    |
        +-------+-----------------------------+
        | S     | secondly                    |
        +-------+-----------------------------+
        | L     | milliseconds                |
        +-------+-----------------------------+
        | U     | microseconds                |
        +-------+-----------------------------+
        | N     | nanoseconds                 |
        +-------+-----------------------------+

        Weekly has the following anchored frequencies:

        +-------+-------------------------------+
        | Alias | Description                   |
        +=======+===============================+
        | W-SUN | weekly frequency (sundays).   |
        |       | Same as 'W'.                  |
        +-------+-------------------------------+
        | W-MON | weekly frequency (mondays)    |
        +-------+-------------------------------+
        | W-TUE | weekly frequency (tuesdays)   |
        +-------+-------------------------------+
        | W-WED | weekly frequency (wednesdays) |
        +-------+-------------------------------+
        | W-THU | weekly frequency (thursdays)  |
        +-------+-------------------------------+
        | W-FRI | weekly frequency (fridays)    |
        +-------+-------------------------------+
        | W-SAT | weekly frequency (saturdays)  |
        +-------+-------------------------------+

        Quarterly frequencies (Q, BQ, QS, BQS) and annual frequencies
        (A, BA, AS, BAS) have the following anchoring suffixes:

        +-------+-------------------------------+
        | Alias | Description                   |
        +=======+===============================+
        | -DEC  | year ends in December (same   |
        |       | as 'Q' and 'A')               |
        +-------+-------------------------------+
        | -JAN  | year ends in January          |
        +-------+-------------------------------+
        | -FEB  | year ends in February         |
        +-------+-------------------------------+
        | -MAR  | year ends in March            |
        +-------+-------------------------------+
        | -APR  | year ends in April            |
        +-------+-------------------------------+
        | -MAY  | year ends in May              |
        +-------+-------------------------------+
        | -JUN  | year ends in June             |
        +-------+-------------------------------+
        | -JUL  | year ends in July             |
        +-------+-------------------------------+
        | -AUG  | year ends in August           |
        +-------+-------------------------------+
        | -SEP  | year ends in September        |
        +-------+-------------------------------+
        | -OCT  | year ends in October          |
        +-------+-------------------------------+
        | -NOV  | year ends in November         |
        +-------+-------------------------------+

        There are some deprecated aggregation interval names in
        tstoolbox, DON'T USE!

        +---------------+-----+
        | Instead of... | Use |
        +===============+=====+
        | hourly        | H   |
        +---------------+-----+
        | daily         | D   |
        +---------------+-----+
        | monthly       | M   |
        +---------------+-----+
        | yearly        | A   |
        +---------------+-----+

        Defaults to D (daily).
    ninterval : int
        The number of agg_interval to use for the aggregation.  Defaults
        to 1.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {round_index}
    {print_input}

    """
    aggd = {'hourly': 'H',
            'daily': 'D',
            'monthly': 'M',
            'yearly': 'A'}
    agg_interval = aggd.get(agg_interval, agg_interval)

    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)
    methods = statistic.split(',')
    newts = pd.DataFrame()
    for method in methods:
        if method == 'mean':
            tmptsd = tsd.resample('{0:d}{1}'.format(ninterval,
                                                    agg_interval)).mean()
        elif method == 'sum':
            tmptsd = tsd.resample('{0:d}{1}'.format(ninterval,
                                                    agg_interval)).sum()
        elif method == 'std':
            tmptsd = tsd.resample('{0:d}{1}'.format(ninterval,
                                                    agg_interval)).std()
        elif method == 'max':
            tmptsd = tsd.resample('{0:d}{1}'.format(ninterval,
                                                    agg_interval)).max()
        elif method == 'min':
            tmptsd = tsd.resample('{0:d}{1}'.format(ninterval,
                                                    agg_interval)).min()
        elif method == 'median':
            tmptsd = tsd.resample('{0:d}{1}'.format(ninterval,
                                                    agg_interval)).median()
        elif method == 'first':
            tmptsd = tsd.resample('{0:d}{1}'.format(ninterval,
                                                    agg_interval)).first()
        elif method == 'last':
            tmptsd = tsd.resample('{0:d}{1}'.format(ninterval,
                                                    agg_interval)).last()
        else:
            raise ValueError("""
***
*** The statistic option must be one of 'mean', 'sum', 'std', 'max',
*** 'min', 'median', 'first', or 'last' to calculate the aggregation.
*** You gave {0}.
***
""".format(statistic))
        tmptsd.rename(columns=lambda x: x + '_' + method, inplace=True)
        newts = newts.join(tmptsd, how='outer')
    return tsutils.print_input(print_input, tsd, newts, '')


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def replace(from_values,
            to_values,
            round_index=None,
            input_ts='-',
            columns=None,
            start_date=None,
            end_date=None,
            dropna='no',
            print_input=False):
    """Return a time-series replacing values with others.

    Parameters
    ----------
    from_values
        All values in this comma separated list are replaced
        with the corresponding value in to_values.  Use the
        string 'None' to represent a missing value.  If
        using 'None' as a from_value it might be easier to
        use the "fill" subcommand instead.
    to_values
        All values in this comma separater list are the
        replacement values corresponding one-to-one to the
        items in from_values.  Use the string 'None' to
        represent a missing value.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {round_index}
    {print_input}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)

    nfrom_values = []
    for fv in from_values.split(','):
        if fv == 'None':
            nfrom_values.append(None)
            continue
        try:
            nfrom_values.append(int(fv))
        except ValueError:
            nfrom_values.append(float(fv))

    nto_values = []
    for tv in to_values.split(','):
        if tv == 'None':
            nto_values.append(None)
            continue
        try:
            nto_values.append(int(tv))
        except ValueError:
            nto_values.append(float(tv))

    ntsd = tsd.replace(nfrom_values, nto_values)
    if dropna in ['any', 'all']:
        ntsd = ntsd.dropna(axis='index', how=dropna)

    return tsutils.print_input(
        print_input, tsd, ntsd, '_replace')


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def clip(input_ts='-',
         start_date=None,
         end_date=None,
         columns=None,
         dropna='no',
         a_min=None,
         a_max=None,
         round_index=None,
         print_input=False):
    """Return a time-series with values limited to [a_min, a_max].

    Parameters
    ---------
    a_min
         All values lower than this will be set to this value.
        Default is None.
    a_max
         All values higher than this will be set to this value.
        Default is None.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {print_input}
    {round_index}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)
    for col in tsd.columns:
        if a_min is None:
            try:
                n_min = pd.np.finfo(tsd[col].dtype).min
            except ValueError:
                n_min = pd.np.iinfo(tsd[col].dtype).min
        else:
            n_min = float(a_min)

        if a_max is None:
            try:
                n_max = pd.np.finfo(tsd[col].dtype).max
            except ValueError:
                n_max = pd.np.iinfo(tsd[col].dtype).max
        else:
            n_max = float(a_max)

    return tsutils.print_input(
        print_input, tsd, tsd.clip(n_min, n_max), '_clip')


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def add_trend(start_offset,
              end_offset,
              input_ts='-',
              columns=None,
              start_date=None,
              end_date=None,
              dropna='no',
              round_index=None,
              print_input=False):
    """Add a trend.

    Parameters
    ----------
    start_offset : float
        The starting value for the applied trend.
    end_offset : float
        The ending value for the applied trend.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {print_input}
    {round_index}
    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)
    # Need it to be float since will be using pd.np.nan
    ntsd = tsd.copy().astype('float64')

    ntsd.ix[:, :] = pd.np.nan
    ntsd.ix[0, :] = float(start_offset)
    ntsd.ix[-1, :] = float(end_offset)
    ntsd = ntsd.interpolate(method='values')

    ntsd = ntsd + tsd

    ntsd = tsutils.memory_optimize(ntsd)
    return tsutils.print_input(print_input,
                               tsd,
                               ntsd,
                               '_trend')


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def remove_trend(input_ts='-',
                 columns=None,
                 start_date=None,
                 end_date=None,
                 dropna='no',
                 round_index=None,
                 print_input=False):
    """Remove a 'trend'.

    Parameters
    ----------
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {round_index}
    {print_input}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)
    ntsd = tsd.copy()
    for col in tsd.columns:
        index = tsd.index.astype('l')
        index = index - index[0]
        lin = pd.np.polyfit(index, tsd[col], 1)
        ntsd[col] = lin[0] * index + lin[1]
        ntsd[col] = tsd[col] - ntsd[col]
    return tsutils.print_input(
        print_input, tsd, ntsd, '_rem_trend')


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def calculate_fdc(input_ts='-',
                  columns=None,
                  start_date=None,
                  end_date=None,
                  percent_point_function=None,
                  plotting_position='weibull',
                  ascending=True):
    """Return the frequency distribution curve.

    DOES NOT return a time-series.

    Parameters
    ----------
    percent_point_function : str
        The distribution used to shift the plotting position values.
        Choose from 'norm', 'lognorm', 'weibull', and None.  Default is
        None.
    plotting_position : str
        'weibull', 'benard', 'tukey', 'gumbel', 'hazen', 'cunnane', or
        'california'.  The default is 'weibull'.

        +------------+-----+-----------------+-----------------------+
        | Name       | a   | Equation        | Description           |
        |            |     | (1-a)/(n+1-2*a) |                       |
        +============+=====+=================+=======================+
        | weibull    | 0   | i/(n+1)         | mean of sampling      |
        |            |     |                 | distribution          |
        |            |     |                 | (default)             |
        +------------+-----+-----------------+-----------------------+
        | benard and | 0.3 | (i-0.3)/(n+0.4) | approx. median of     |
        | bos-       |     |                 | sampling distribution |
        | levenbach  |     |                 |                       |
        +------------+-----+-----------------+-----------------------+
        | tukey      | 1/3 | (i-1/3)/(n+1/3) | approx. median of     |
        |            |     |                 | sampling distribution |
        +------------+-----+-----------------+-----------------------+
        | gumbel     | 1   | (i-1)/(n-1)     | mode of sampling      |
        |            |     |                 | distribution          |
        +------------+-----+-----------------+-----------------------+
        | hazen      | 1/2 | (i-1/2)/n       | midpoints of n equal  |
        |            |     |                 | intervals             |
        +------------+-----+-----------------+-----------------------+
        | cunnane    | 2/5 | (i-2/5)/(n+1/5) | subjective            |
        +------------+-----+-----------------+-----------------------+
        | california | NA  | i/n             |                       |
        +------------+-----+-----------------+-----------------------+

        Where 'i' is the sorted rank of the y value, and 'n' is the
        total number of values to be plotted.
    ascending : bool
        Sort order defaults to True.
    {input_ts}
    {columns}
    {start_date}
    {end_date}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns)

    ppf = _set_ppf(percent_point_function)
    newts = pd.DataFrame()
    for col in tsd:
        tmptsd = tsd[col].dropna()
        xdat = ppf(_set_plotting_position(tmptsd.count(),
                                          plotting_position)) * 100
        tmptsd.sort_values(ascending=ascending, inplace=True)
        tmptsd.index = xdat
        newts = newts.join(tmptsd, how='outer')
    newts.index.name = 'Plotting_position'
    newts = newts.groupby(newts.index).first()
    return tsutils.printiso(newts,
                            showindex="always")


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def stack(input_ts='-',
          columns=None,
          start_date=None,
          end_date=None,
          round_index=None,
          dropna='no'):
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
    {round_index}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)

    newtsd = pd.DataFrame(tsd.stack()).reset_index(1)
    newtsd.columns = ['Columns', 'Values']
    newtsd = newtsd.groupby('Columns').apply(
        lambda d: d.sort_values('Columns')).reset_index('Columns', drop=True)
    return tsutils.printiso(newtsd)


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def unstack(column_names,
            input_ts='-',
            columns=None,
            start_date=None,
            end_date=None,
            round_index=None,
            dropna='no'):
    """Return the unstack of the input table.

    The unstack command takes the stacked table and converts to a
    standard tstoolbox table.

    From::

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

    To::

      Datetime,TS1,TS2,TS3
      2000-01-01,1.2,1018.2,0.0032
      2000-01-02,1.8,1453.1,0.0002
      2000-01-03,1.9,1683.1,-0.0004

    Parameters
    ----------
    column_names
        The column in the table that holds the column names
        of the unstacked data.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {dropna}
    {round_index}

    """
    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna=dropna)

    cols = list(tsd.columns)
    cols.remove(column_names)
    newtsd = pd.DataFrame(tsd[cols].values,
                          index=[tsd.index.values,
                                 tsd[column_names].values])

    try:
        newtsd = newtsd.unstack()
    except ValueError:
        raise ValueError("""
*
*   Duplicate index (time stamp and '{0}') where found.
*   Found these duplicate indices:
{1}
*
""".format(column_names,
           newtsd.index.get_duplicates()))

    newtsd.index.name = 'Datetime'
    levels = newtsd.columns.levels
    labels = newtsd.columns.labels
    newtsd.columns = levels[1][labels[1]]

    # Remove weird characters from column names
    newtsd.rename(columns=lambda x: ''.join(
        [i for i in str(x) if i not in '\'" ']))
    return tsutils.printiso(newtsd)


mark_dict = {
    ".": "point",
    ",": "pixel",
    "o": "circle",
    "v": "triangle_down",
    "^": "triangle_up",
    "<": "triangle_left",
    ">": "triangle_right",
    "1": "tri_down",
    "2": "tri_up",
    "3": "tri_left",
    "4": "tri_right",
    "8": "octagon",
    "s": "square",
    "p": "pentagon",
    "*": "star",
    "h": "hexagon1",
    "H": "hexagon2",
    "+": "plus",
    "D": "diamond",
    "d": "thin_diamond",
    "|": "vline",
    "_": "hline"
}

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k',
          'aliceblue',
          'antiquewhite',
          'aqua',
          'aquamarine',
          'azure',
          'beige',
          'bisque',
          'blanchedalmond',
          'blueviolet',
          'burlywood',
          'cadetblue',
          'chartreuse',
          'chocolate']


def _set_ppf(ptype):
    if ptype == 'norm':
        from scipy.stats.distributions import norm
        return norm.ppf
    elif ptype == 'lognorm':
        from scipy.stats.distributions import lognorm
        return lognorm.freeze(0.5, loc=0).ppf
    elif ptype == 'weibull':
        def ppf(y):
            """Percentage Point Function for the weibull distibution."""
            return pd.np.log(-pd.np.log((1 - pd.np.array(y))))
        return ppf
    elif ptype is None:
        def ppf(y):
            return y
        return ppf


def _plotting_position_equation(i, n, a):
    """ Parameterized, generic plotting position equation."""
    return (i - a) / float(n + 1 - 2 * a)


def _set_plotting_position(n, plotting_position='weibull'):
    """ Create plotting position 1D array using linspace. """
    if plotting_position == 'weibull':
        return pd.np.linspace(_plotting_position_equation(1, n, 0.0),
                              _plotting_position_equation(n, n, 0.0),
                              n)
    elif plotting_position == 'benard':
        return pd.np.linspace(_plotting_position_equation(1, n, 0.3),
                              _plotting_position_equation(n, n, 0.3),
                              n)
    elif plotting_position == 'tukey':
        return pd.np.linspace(_plotting_position_equation(1, n, 1.0 / 3.0),
                              _plotting_position_equation(n, n, 1.0 / 3.0),
                              n)
    elif plotting_position == 'gumbel':
        return pd.np.linspace(_plotting_position_equation(1, n, 1.0),
                              _plotting_position_equation(n, n, 1.0),
                              n)
    elif plotting_position == 'hazen':
        return pd.np.linspace(_plotting_position_equation(1, n, 1.0 / 2.0),
                              _plotting_position_equation(n, n, 1.0 / 2.0),
                              n)
    elif plotting_position == 'cunnane':
        return pd.np.linspace(_plotting_position_equation(1, n, 2.0 / 5.0),
                              _plotting_position_equation(n, n, 2.0 / 5.0),
                              n)
    elif plotting_position == 'california':
        return pd.np.linspace(1. / n, 1., n)
    else:
        raise ValueError("""
*
*    The plotting_position option accepts 'weibull', 'benard', 'tukey',
*    'gumbel', 'hazen', 'cunnane', and 'california'
*    plotting position options, you gave {0}.
*
""".format(plotting_position))


@mando.command(formatter_class=RSTHelpFormatter, doctype='numpy')
@tsutils.doc(tsutils.docstrings)
def plot(input_ts='-',
         columns=None,
         start_date=None,
         end_date=None,
         ofilename='plot.png',
         type='time',
         xtitle='',
         ytitle='',
         title='',
         figsize='10,6.0',
         legend=None,
         legend_names=None,
         subplots=False,
         sharex=True,
         sharey=False,
         style=None,
         logx=False,
         logy=False,
         xaxis='arithmetic',
         yaxis='arithmetic',
         xlim=None,
         ylim=None,
         secondary_y=False,
         mark_right=True,
         scatter_matrix_diagonal='probability_density',
         bootstrap_size=50,
         bootstrap_samples=500,
         norm_xaxis=False,
         norm_yaxis=False,
         lognorm_xaxis=False,
         lognorm_yaxis=False,
         xy_match_line='',
         grid=None,
         label_rotation=None,
         label_skip=1,
         force_freq=None,
         drawstyle='default',
         por=False,
         invert_xaxis=False,
         invert_yaxis=False,
         round_index=None,
         plotting_position='weibull'):
    """Plot data.

    Parameters
    ----------
    ofilename : str
        Output filename for the plot.  Extension defines
        the type, ('.png'). Defaults to 'plot.png'.
    type : str
        The plot type.  Defaults to 'time'.

        Can be one of the following:

        time
            standard time series plot
        xy
            (x,y) plot, also know as a scatter plot
        double_mass
            (x,y) plot of the cumulative sum of x and y
        boxplot
            box extends from lower to upper quartile, with line at the
            median.  Depending on the statistics, the wiskers represent
            the range of the data or 1.5 times the inter-quartile range
            (Q3 - Q1)
        scatter_matrix
            plots all columns against each other
        lag_plot
            indicates structure in the data
        autocorrelation
            plot autocorrelation
        bootstrap
            visually asses aspects of a data set by plotting random
            selections of values
        probability_density
            sometime called kernel density estimation (KDE)
        bar
            sometimes called a column plot
        barh
            a horizontal bar plot
        bar_stacked
            sometimes called a stacked column
        barh_stacked
            a horizontal stacked bar plot
        histogram
            calculate and create a histogram plot
        norm_xaxis
            sort, calculate probabilities, and plot data against an
            x axis normal distribution
        norm_yaxis
            sort, calculate probabilities, and plot data against an
            y axis normal distribution
        lognorm_xaxis
            sort, calculate probabilities, and plot data against an
            x axis lognormal distribution
        lognorm_yaxis
            sort, calculate probabilities, and plot data against an
            y axis lognormal distribution
        weibull_xaxis
            sort, calculate and plot data against an x axis weibull
            distribution
        weibull_yaxis
            sort, calculate and plot data against an y axis weibull
            distribution
    xtitle : str
        Title of x-axis, default depend on ``type``.
    ytitle : str
        Title of y-axis, default depend on ``type``.
    title : str
        Title of chart, defaults to ''.
    figsize : str
        The 'width,height' of plot as inches.
        Defaults to '10,6.5'.
    legend
        Whether to display the legend. Defaults to True.
    legend_names : str
        Legend would normally use the time-series names
        associated with the input data.  The 'legend_names' option allows you
        to override the names in the data set.  You must supply a comma
        separated list of strings for each time-series in the data set.
        Defaults to None.
    subplots
        boolean, default False.  Make separate subplots for
        each time series
    sharex
        boolean, default True In case subplots=True, share
        x axis
    sharey
        boolean, default False In case subplots=True, share y axis
    style : str
        Comma separated matplotlib style strings matplotlib
        line style per time-series.  Just combine codes in 'ColorLineMarker'
        order, for example r--* is a red dashed line with star marker.

        +------+---------+
        | Code | Color   |
        +======+=========+
        | b    | blue    |
        +------+---------+
        | g    | green   |
        +------+---------+
        | r    | red     |
        +------+---------+
        | c    | cyan    |
        +------+---------+
        | m    | magenta |
        +------+---------+
        | y    | yellow  |
        +------+---------+
        | k    | black   |
        +------+---------+
        | w    | white   |
        +------+---------+

        +---------+-----------+
        | Number  | Color     |
        +=========+===========+
        | 0.75    | 0.75 gray |
        +---------+-----------+
        | ...etc. |           |
        +---------+-----------+

        +------------------+
        | HTML Color Names |
        +==================+
        | red              |
        +------------------+
        | burlywood        |
        +------------------+
        | chartreuse       |
        +------------------+
        | ...etc.          |
        +------------------+

        Color reference:
        http://matplotlib.org/api/colors_api.html

        +------+--------------+
        | Code | Lines        |
        +======+==============+
        | -    | solid        |
        +------+--------------+
        | --   | dashed       |
        +------+--------------+
        | -.   | dash_dot     |
        +------+--------------+
        | :    | dotted       |
        +------+--------------+
        | None | draw nothing |
        +------+--------------+
        | ' '  | draw nothing |
        +------+--------------+
        | ''   | draw nothing |
        +------+--------------+

        Line reference:
        http://matplotlib.org/api/artist_api.html

        +------+----------------+
        | Code | Markers        |
        +======+================+
        | .    | point          |
        +------+----------------+
        | o    | circle         |
        +------+----------------+
        | v    | triangle down  |
        +------+----------------+
        | ^    | triangle up    |
        +------+----------------+
        | <    | triangle left  |
        +------+----------------+
        | >    | triangle right |
        +------+----------------+
        | 1    | tri_down       |
        +------+----------------+
        | 2    | tri_up         |
        +------+----------------+
        | 3    | tri_left       |
        +------+----------------+
        | 4    | tri_right      |
        +------+----------------+
        | 8    | octagon        |
        +------+----------------+
        | s    | square         |
        +------+----------------+
        | p    | pentagon       |
        +------+----------------+
        | *    | star           |
        +------+----------------+
        | h    | hexagon1       |
        +------+----------------+
        | H    | hexagon2       |
        +------+----------------+
        | +    | plus           |
        +------+----------------+
        | x    | x              |
        +------+----------------+
        | D    | diamond        |
        +------+----------------+
        | d    | thin diamond   |
        +------+----------------+
        | _    | hline          |
        +------+----------------+
        | None | nothing        |
        +------+----------------+
        | ' '  | nothing        |
        +------+----------------+
        | ''   | nothing        |
        +------+----------------+

        Marker reference:
        http://matplotlib.org/api/markers_api.html

    logx
        DEPRECATED: use '--xaxis="log"' instead.
    logy
        DEPRECATED: use '--yaxis="log"' instead.
    xlim
        Comma separated lower and upper limits (--xlim 1,1000) Limits
        for the x-axis.  Default is based on range of x values.
    ylim
        Comma separated lower and upper limits (--ylim 1,1000) Limits
        for the y-axis.  Default is based on range of y values.
    xaxis : str
        Defines the type of the xaxis.  One of 'arithmetic',
        'log'. Default is 'arithmetic'.
    yaxis : str
        Defines the type of the yaxis.  One of 'arithmetic',
        'log'. Default is 'arithmetic'.
    secondary_y
        Boolean or sequence, default False Whether to plot on
        the secondary y-axis If a list/tuple, which time-series to plot on
        secondary y-axis
    mark_right
        Boolean, default True : When using a secondary_y
        axis, should the legend label the axis of the various time-series
        automatically
    scatter_matrix_diagonal : str
        If plot type is 'scatter_matrix',
        this specifies the plot along the diagonal.  Defaults to
        'probability_density'.
    bootstrap_size
        The size of the random subset for 'bootstrap' plot.
        Defaults to 50.
    bootstrap_samples
        The number of random subsets of
        'bootstrap_size'.  Defaults to 500.
    norm_xaxis
        DEPRECATED: use '--type="norm_xaxis"' instead.
    norm_yaxis
        DEPRECATED: use '--type="norm_yaxis"' instead.
    lognorm_xaxis
        DEPRECATED: use '--type="lognorm_xaxis"' instead.
    lognorm_yaxis
        DEPRECATED: use '--type="lognorm_yaxis"' instead.
    xy_match_line : str
        Will add a match line where x == y.  Default
        is ''.  Set to a line style code.
    grid
        Boolean, default True Whether to plot grid lines on the major
        ticks.
    label_rotation : int
        Rotation for major labels for bar plots.
    label_skip : int
        Skip for major labels for bar plots.
    drawstyle : str
        'default' connects the points with lines. The
        steps variants produce step-plots. 'steps' is equivalent to 'steps-pre'
        and is maintained for backward-compatibility.

        ACCEPTS::

         ['default' | 'steps' | 'steps-pre' | 'steps-mid' | 'steps-post']

    por
        Plot from first good value to last good value.  Strip NANs
        from beginning and end.
    force_freq
        Force this frequency for the plot.  WARNING: you may
        lose data if not careful with this option.  In general, letting the
        algorithm determine the frequency should always work, but this option
        will override.  Use PANDAS offset codes,
    invert_xaxis
        Invert the x-axis.
    invert_yaxis
        Invert the y-axis.
    plotting_position : str
        'weibull', 'benard', 'tukey', 'gumbel',
        'hazen', 'cunnane', or 'california'.  The default is 'weibull'.

        +------------+-----+-----------------+-----------------------+
        | Name       | a   | Equation        | Description           |
        |            |     | (1-a)/(n+1-2*a) |                       |
        +============+=====+=================+=======================+
        | weibull    | 0   | i/(n+1)         | mean of sampling      |
        |            |     |                 | distribution          |
        |            |     |                 | (default)             |
        +------------+-----+-----------------+-----------------------+
        | benard and | 0.3 | (i-0.3)/(n+0.4) | approx. median of     |
        | bos-       |     |                 | sampling distribution |
        | levenbach  |     |                 |                       |
        +------------+-----+-----------------+-----------------------+
        | tukey      | 1/3 | (i-1/3)/(n+1/3) | approx. median of     |
        |            |     |                 | sampling distribution |
        +------------+-----+-----------------+-----------------------+
        | gumbel     | 1   | (i-1)/(n-1)     | mode of sampling      |
        |            |     |                 | distribution          |
        +------------+-----+-----------------+-----------------------+
        | hazen      | 1/2 | (i-1/2)/n       | midpoints of n equal  |
        |            |     |                 | intervals             |
        +------------+-----+-----------------+-----------------------+
        | cunnane    | 2/5 | (i-2/5)/(n+1/5) | subjective            |
        +------------+-----+-----------------+-----------------------+
        | california | NA  | i/n             |                       |
        +------------+-----+-----------------+-----------------------+

        Where 'i' is the sorted rank of the y value, and 'n'
        is the total number of values to be plotted.

        Only used for norm_xaxis, norm_yaxis, lognorm_xaxis,
        lognorm_yaxis, weibull_xaxis, and weibull_yaxis.
    {input_ts}
    {columns}
    {start_date}
    {end_date}
    {round_index}
    """
    # Need to work around some old option defaults with the implementation of
    # mando
    legend = bool(legend == '' or legend == 'True' or legend is None)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator

    tsd = tsutils.common_kwds(tsutils.read_iso_ts(input_ts),
                              start_date=start_date,
                              end_date=end_date,
                              pick=columns,
                              round_index=round_index,
                              dropna='all')

    if por is True:
        tsd = tsutils.common_kwds(tsutils.read_iso_ts(tsd),
                                  start_date=start_date,
                                  end_date=end_date,
                                  round_index=round_index,
                                  dropna='no')

    def _know_your_limits(xylimits, axis='arithmetic'):
        """Establish axis limits.

        This defines the xlim and ylim as lists rather than strings.
        Might prove useful in the future in a more generic spot.  It
        normalizes the different representations.
        """
        if isinstance(xylimits, str):
            nlim = []
            for lim in xylimits.split(','):
                if lim == '':
                    nlim.append(None)
                elif '.' in lim:
                    nlim.append(float(lim))
                else:
                    nlim.append(int(lim))
        else:  # tuples or lists...
            nlim = xylimits

        if axis in ['norm', 'lognormal', 'weibull']:
            if nlim is None:
                nlim = [None, None]
            if nlim[0] is None:
                nlim[0] = 0.01
            if nlim[1] is None:
                nlim[1] = 0.99
            if (nlim[0] <= 0 or nlim[0] >= 1 or
                    nlim[1] <= 0 or nlim[1] >= 1):
                raise ValueError("""
*
*   Both limits must be between 0 and 1 for the
*   'normal', 'lognormal', or 'weibull' axis.
*
*   Instead you have {0}.
*
""".format(nlim))

        if nlim is None:
            return nlim

        if nlim[0] is not None and nlim[1] is not None:
            if nlim[0] >= nlim[1]:
                raise ValueError("""
*
*   The second limit must be greater than the first.
*
*   You gave {0}.
*
""".format(nlim))

        if axis == 'log':
            if ((nlim[0] is not None and nlim[0] <= 0) or
                    (nlim[1] is not None and nlim[1] <= 0)):
                raise ValueError("""
*
*   If log plot cannot have limits less than or equal to 0.
*
*   You have {0}.
*
""".format(nlim))

        return nlim

    # This is to help pretty print the frequency
    try:
        try:
            pltfreq = str(tsd.index.freq, 'utf-8').lower()
        except TypeError:
            pltfreq = str(tsd.index.freq).lower()
        if pltfreq.split(' ')[0][1:] == '1':
            beginstr = 3
        else:
            beginstr = 1
        # short freq string (day) OR (2 day)
        short_freq = '({0})'.format(pltfreq[beginstr:-1])
    except AttributeError:
        short_freq = ''

    if legend_names:
        lnames = legend_names.split(',')
        if len(lnames) != len(set(lnames)):
            raise ValueError("""
*
*   Each name in legend_names must be unique.
*
""")
        if len(tsd.columns) == len(lnames):
            renamedict = dict(zip(tsd.columns, lnames))
        elif type == 'xy' and len(tsd.columns) // 2 == len(lnames):
            renamedict = dict(zip(tsd.columns[2::2], lnames[1:]))
            renamedict[tsd.columns[1]] = lnames[0]
        else:
            raise ValueError("""
*
*   For 'legend_names' you must have the same number of comma
*   separated names as columns in the input data.  The input
*   data has {0} where the number of 'legend_names' is {1}.
*
*   If 'xy' type you need to have legend names as x,y1,y2,y3,...
*
""".format(len(tsd.columns), len(lnames)))
        tsd.rename(columns=renamedict, inplace=True)
    else:
        lnames = tsd.columns

    if style:
        style = style.split(',')

    if (logx is True or
            logy is True or
            norm_xaxis is True or
            norm_yaxis is True or
            lognorm_xaxis is True or
            lognorm_yaxis is True):
        warnings.warn("""
*
*   The --logx, --logy, --norm_xaxis, --norm_yaxis, --lognorm_xaxis, and
*   --lognorm_yaxis options are deprecated.
*
*   For --logx use --xaxis="log"
*   For --logy use --yaxis="log"
*   For --norm_xaxis use --type="norm_xaxis"
*   For --norm_yaxis use --type="norm_yaxis"
*   For --lognorm_xaxis use --type="lognorm_xaxis"
*   For --lognorm_yaxis use --type="lognorm_yaxis"
*
""")

    if xaxis == 'log':
        logx = True
    if yaxis == 'log':
        logy = True

    if type in ['norm_xaxis', 'lognorm_xaxis', 'weibull_xaxis']:
        xaxis = 'normal'
        if logx is True:
            logx = False
            warnings.warn("""
*
*   The --type={1} cannot also have the xaxis set to {0}.
*   The {0} setting for xaxis is ignored.
*
""".format(xaxis, type))

    if type in ['norm_yaxis', 'lognorm_yaxis', 'weibull_yaxis']:
        yaxis = 'normal'
        if logy is True:
            logy = False
            warnings.warn("""
*
*   The --type={1} cannot also have the yaxis set to {0}.
*   The {0} setting for yaxis is ignored.
*
""".format(yaxis, type))

    xlim = _know_your_limits(xlim, axis=xaxis)
    ylim = _know_your_limits(ylim, axis=yaxis)

    figsize = [float(i) for i in figsize.split(',')]
    plt.figure(figsize=figsize)

    if not isinstance(tsd.index, pd.DatetimeIndex):
        tsd.insert(0, tsd.index.name, tsd.index)

    if type == 'time':
        tsd.plot(legend=legend, subplots=subplots, sharex=sharex,
                 sharey=sharey, style=style, logx=logx, logy=logy, xlim=xlim,
                 ylim=ylim, secondary_y=secondary_y, mark_right=mark_right,
                 figsize=figsize, drawstyle=drawstyle)
        plt.xlabel(xtitle or 'Time')
        plt.ylabel(ytitle)
        if legend is True:
            plt.legend(loc='best')
    elif type in ['xy',
                  'double_mass',
                  'norm_xaxis',
                  'norm_yaxis',
                  'lognorm_xaxis',
                  'lognorm_yaxis',
                  'weibull_xaxis',
                  'weibull_yaxis']:
        # PANDAS was not doing the right thing with xy plots
        # if you wanted lines between markers.
        # Fell back to using raw matplotlib.
        # Boy I do not like matplotlib.
        _, ax = plt.subplots(figsize=figsize)

        # Make a default set of 'style' strings if 'style' is None.
        if style is None:
            typed = '.-'
            if type in ['xy']:
                typed = '*'
            style = zip(colors * (len(tsd.columns) // len(colors) + 1),
                        [typed] * len(tsd.columns))
            style = [i + j for i, j in style]

        if type == 'double_mass':
            tsd = tsd.cumsum()

        if type in ['norm_xaxis',
                    'norm_yaxis',
                    'lognorm_xaxis',
                    'lognorm_yaxis',
                    'weibull_xaxis',
                    'weibull_yaxis']:
            ppf = _set_ppf(type.split('_')[0])
            ys = tsd.iloc[:, :]
            colcnt = tsd.shape[1]
        else:
            xs = pd.np.array(tsd.iloc[:, 0::2])
            ys = pd.np.array(tsd.iloc[:, 1::2])
            colcnt = tsd.shape[1] // 2

        for colindex in range(colcnt):
            lstyle = style[colindex]
            lcolor = 'b'
            marker = ''
            linest = '-'
            for mdict in mark_dict:
                if mdict == lstyle[-1]:
                    marker = mdict
                    lstyle = lstyle.rstrip(mdict)
                    break
            for l in ["--", "-", "-.", ":", " "]:
                if l in lstyle:
                    linest = l
                    lstyle = lstyle.rstrip(l)
                    break
            lcolor = lstyle

            if type in ['norm_xaxis', 'norm_yaxis',
                        'lognorm_xaxis', 'lognorm_yaxis',
                        'weibull_xaxis', 'weibull_yaxis']:
                oydata = pd.np.array(ys.iloc[:, colindex].dropna())
                oydata = pd.np.sort(oydata)[::-1]
                n = len(oydata)
                norm_axis = ax.xaxis
                oxdata = ppf(_set_plotting_position(n, plotting_position))
            else:
                oxdata = xs[:, colindex]
                oydata = ys[:, colindex]

            if type in ['norm_yaxis', 'lognorm_yaxis', 'weibull_yaxis']:
                oxdata, oydata = oydata, oxdata
                norm_axis = ax.yaxis

            # Make the plot for each column
            if logy is True and logx is False:
                ax.semilogy(oxdata, oydata,
                            linestyle=linest,
                            color=lcolor,
                            marker=marker,
                            label=lnames[colindex])
            elif logx is True and logy is False:
                ax.semilogx(oxdata, oydata,
                            linestyle=linest,
                            color=lcolor,
                            marker=marker,
                            label=lnames[colindex])
            elif logx is True and logy is True:
                ax.loglog(oxdata, oydata,
                          linestyle=linest,
                          color=lcolor,
                          marker=marker,
                          label=lnames[colindex])
            else:
                ax.plot(oxdata, oydata,
                        linestyle=linest,
                        color=lcolor,
                        marker=marker,
                        label=lnames[colindex],
                        drawstyle=drawstyle)

        # Make it pretty
        if type in ['norm_xaxis', 'norm_yaxis',
                    'lognorm_xaxis', 'lognorm_yaxis',
                    'weibull_xaxis', 'weibull_yaxis']:
            xtmaj = pd.np.array([0.01, 0.1, 0.5, 0.9, 0.99])
            xtmaj_str = ['1', '10', '50', '90', '99']
            xtmin = pd.np.concatenate([pd.np.linspace(0.001, 0.01, 10),
                                       pd.np.linspace(0.01, 0.1, 10),
                                       pd.np.linspace(0.1, 0.9, 9),
                                       pd.np.linspace(0.9, 0.99, 10),
                                       pd.np.linspace(0.99, 0.999, 10),
                                       ])
            xtmaj = ppf(xtmaj)
            xtmin = ppf(xtmin)

            norm_axis.set_major_locator(FixedLocator(xtmaj))
            norm_axis.set_minor_locator(FixedLocator(xtmin))

        if type in ['norm_xaxis', 'lognorm_xaxis', 'weibull_xaxis']:
            ax.set_xticklabels(xtmaj_str)
            ax.set_ylim(ylim)
            ax.set_xlim(ppf([0.01, 0.99]))

        elif type in ['norm_yaxis', 'lognorm_yaxis', 'weibull_yaxis']:
            ax.set_yticklabels(xtmaj_str)
            ax.set_xlim(xlim)
            ax.set_ylim(ppf(ylim))
        else:
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)

        if xy_match_line:
            if isinstance(xy_match_line, str):
                xymsty = xy_match_line
            else:
                xymsty = 'g--'
            nxlim = ax.get_xlim()
            nylim = ax.get_ylim()
            maxt = max(nxlim[1], nylim[1])
            mint = min(nxlim[0], nylim[0])
            ax.plot([mint, maxt], [mint, maxt], xymsty, zorder=1)
            ax.set_ylim(nylim)
            ax.set_xlim(nxlim)

        if type in ['xy', 'double_mass']:
            xtitle = xtitle or tsd.columns[0]
            ytitle = ytitle or tsd.columns[1]
        elif type in ['norm_xaxis']:
            xtitle = xtitle or 'Normal Distribution'
            ytitle = ytitle or tsd.columns[0]
        elif type in ['lognorm_xaxis']:
            xtitle = xtitle or 'Log Normal Distribution'
            ytitle = ytitle or tsd.columns[0]
        elif type in ['weibull_xaxis']:
            xtitle = xtitle or 'Weibull Distribution'
            ytitle = ytitle or tsd.columns[0]
        if type in ['norm_yaxis', 'weibull_yaxis']:
            xtitle, ytitle = ytitle, xtitle

        ax.set_xlabel(xtitle or tsd.columns[0])
        ax.set_ylabel(ytitle or tsd.columns[1])
        if legend is True:
            ax.legend(loc='best')
    elif type == 'probability_density':
        tsd.plot(kind='kde', legend=legend, subplots=subplots, sharex=sharex,
                 sharey=sharey, style=style, logx=logx, logy=logy, xlim=xlim,
                 ylim=ylim, secondary_y=secondary_y,
                 figsize=figsize)
        plt.xlabel(xtitle)
        plt.ylabel(ytitle or 'Density')
        if legend is True:
            plt.legend(loc='best')
    elif type == 'boxplot':
        tsd.boxplot()
    elif type == 'scatter_matrix':
        from pandas.tools.plotting import scatter_matrix
        if scatter_matrix_diagonal == 'probablity_density':
            scatter_matrix_diagonal = 'kde'
        scatter_matrix(tsd, diagonal=scatter_matrix_diagonal,
                       figsize=figsize)
    elif type == 'lag_plot':
        from pandas.tools.plotting import lag_plot
        lag_plot(tsd,
                 figsize=figsize)
        plt.xlabel(xtitle or 'y(t)')
        plt.ylabel(ytitle or 'y(t+{0})'.format(short_freq or 1))
    elif type == 'autocorrelation':
        from pandas.tools.plotting import autocorrelation_plot
        autocorrelation_plot(tsd,
                             figsize=figsize)
        plt.xlabel(xtitle or 'Time Lag {0}'.format(short_freq))
        plt.ylabel(ytitle)
    elif type == 'bootstrap':
        if len(tsd.columns) > 1:
            raise ValueError("""
*
*   The 'bootstrap' plot can only work with 1 time-series in the DataFrame.
*   The DataFrame that you supplied has {0} time-series.
*
""".format(len(tsd.columns)))
        from pandas.tools.plotting import bootstrap_plot
        bootstrap_plot(tsd, size=bootstrap_size, samples=bootstrap_samples,
                       color='gray',
                       figsize=figsize)
    elif (type == 'bar' or
          type == 'bar_stacked' or
          type == 'barh' or
          type == 'barh_stacked'):
        stacked = False
        if type[-7:] == 'stacked':
            stacked = True
        kind = 'bar'
        if type[:4] == 'barh':
            kind = 'barh'
        ax = tsd.plot(kind=kind, legend=legend, stacked=stacked,
                      style=style, logx=logx, logy=logy, xlim=xlim,
                      ylim=ylim,
                      figsize=figsize)
        freq = tsutils.asbestfreq(tsd, force_freq=force_freq)[1]
        if freq is not None:
            if 'A' in freq:
                endchar = 4
            elif 'M' in freq:
                endchar = 7
            elif 'D' in freq:
                endchar = 10
            elif 'H' in freq:
                endchar = 13
            else:
                endchar = None
            nticklabels = []
            if kind == 'bar':
                taxis = ax.xaxis
            else:
                taxis = ax.yaxis
            for index, i in enumerate(taxis.get_majorticklabels()):
                if index % label_skip:
                    nticklabels.append(' ')
                else:
                    nticklabels.append(i.get_text()[:endchar])
            taxis.set_ticklabels(nticklabels)
            plt.setp(taxis.get_majorticklabels(), rotation=label_rotation)
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        if legend is True:
            plt.legend(loc='best')
    elif type == 'histogram':
        tsd.hist(figsize=figsize)
        plt.xlabel(xtitle)
        plt.ylabel(ytitle)
        if legend is True:
            plt.legend(loc='best')
    else:
        raise ValueError("""
*
*   Plot 'type' {0} is not supported.
*
""".format(type))

    grid = bool(grid is None)
    if invert_xaxis is True:
        plt.gca().invert_xaxis()
    if invert_yaxis is True:
        plt.gca().invert_yaxis()
    plt.grid(grid)
    plt.title(title)
    plt.tight_layout()
    if ofilename is None:
        return plt
    plt.savefig(ofilename)


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
    window
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
        The number of groups to separate the
        time series into.
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
        'minmax', 'zscore', or 'pct_rank'.  Default is
        'minmax'

        minmax
            min_limit +
            (X-Xmin)/(Xmax-Xmin)*(max_limit-min_limit)

        zscore
            X-mean(X)/stddev(X)

        pct_rank
            rank(X)*100/N
    min_limit : float
        Defaults to 0.  Defines the minimum limit
        of the minmax normalization.
    max_limit : float
        Defaults to 1.  Defines the maximum limit
        of the minmax normalization.
    pct_rank_method : str
        Defaults to 'average'.  Defines how
        tied ranks are broken.  Can be 'average', 'min', 'max', 'first',
        'dense'.
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
                            showindex="always")


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
        Can be one of, 'julian' (the default), 'reduced', 'modified',
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
                            showindex="always")


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
        The number of intervals to calculate percent
        change across.
    fill_method : str
        Fill method for NA.  Defaults to 'pad'.
    limit
        Defaults to None.  Is the minimum number of
        consecutive NA values where no more filling will be made.
    freq : str
        A pandas time offset string to represent the
        interval.
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
        [0 or 'index' or 1 or 'columns'], default 0.
        Index to direct ranking
    method : str
        ['average', 'min', 'max', 'first', 'dense'], default
        'average'.

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
        boolean, default None
        Include only float, int, boolean data. Valid only for DataFrame or
        Panel objects
    na_option : str
        ['keep', 'top', 'bottom'], default is 'keep'.

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
        boolean, default True.
        False for ranks by high (1) to low (N)
    pct
        boolean, default False.
        Computes percentage rank of data
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
                dropna="no",
                input_ts="-",
                start_date=None,
                end_date=None,
                round_index=None):
    """Apply an offset to a time-series.

    Parameters
    ----------
    years: number
        Relative number of years to offset the datetime index,
        may be negative; adding or subtracting a relativedelta with
        relative information performs the corresponding arithmetic
        operation on the original datetime value with the information in
    months: number
        Relative number of months to offset the datetime index,
        may be negative; adding or subtracting a relativedelta with
        relative information performs the corresponding arithmetic
        operation on the original datetime value with the information in
    weeks: number
        Relative number of weeks to offset the datetime index,
        may be negative; adding or subtracting a relativedelta with
        relative information performs the corresponding arithmetic
        operation on the original datetime value with the information in
    days: number
        Relative number of days to offset the datetime index,
        may be negative; adding or subtracting a relativedelta with
        relative information performs the corresponding arithmetic
        operation on the original datetime value with the information in
    hours: number
        Relative number of hours to offset the datetime index,
        may be negative; adding or subtracting a relativedelta with
        relative information performs the corresponding arithmetic
        operation on the original datetime value with the information in
    minutes: number
        Relative number of minutes to offset the datetime index,
        may be negative; adding or subtracting a relativedelta with
        relative information performs the corresponding arithmetic
        operation on the original datetime value with the information in
    seconds: number
        Relative number of seconds to offset the datetime index,
        may be negative; adding or subtracting a relativedelta with
        relative information performs the corresponding arithmetic
        operation on the original datetime value with the information in
    microseconds: number
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

    ntsd = pd.DataFrame(tsd.values,
                        index=[i +
                               pd.tseries.offsets.relativedelta(years=years,
                                                                months=months,
                                                                days=days,
                                                                hours=hours,
                                                                minutes=minutes,
                                                                seconds=seconds,
                                                                microseconds=microseconds)
                               for i in tsd.index])
    ntsd.columns = tsd.columns

    return tsutils.printiso(ntsd, showindex="always")


def main():
    """Main function."""
    if not os.path.exists('debug_tstoolbox'):
        sys.tracebacklimit = 0
    mando.main()


if __name__ == '__main__':
    main()
