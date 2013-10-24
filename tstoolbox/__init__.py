#!/sjr/beodata/local/python_linux/bin/python

from __future__ import print_function

'''
tstoolbox is a collection of command line tools for the manipulation of time
series.
'''
import sys
import os.path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from numpy import *
import baker

import tsutils
from tstoolbox.fill_functions import fill

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


def _date_slice(input_ts='-', start_date=None, end_date=None):
    '''
    Private function to slice time series.
    '''
    tsd = tsutils.read_iso_ts(input_ts)
    return tsd[start_date:end_date]


def _sniff_filetype(filename):

    # Is it a pickled file...
    try:
        return pd.core.common.load(filename)
    except:
        pass

    # Really hard to determine if there is a header -
    # Assume yes...
    return pd.read_table(open(filename, 'rb'), header=0, sep=',',
                         parse_dates=[0], index_col=[0])


@baker.command
def filter(filter_type, print_input=False, start_freq=30, end_freq=40, window_len=5, input_ts='-'):
    '''
    Apply different filters to the time-series.

    :param filter_type: 'fft_highpass' and 'fft_lowpass' for Fast Fourier
        Transform filter in the frequency domain.
    :param print_input: If set to 'True' will include the input columns in the
        output table.  Default is 'False'.
    :param input_ts: Filename with data in 'ISOdate,value' format or '-' for
        stdin.  Default is stdin.
    :param start_freq: For 'fft_highpass' and 'fft_lowpass'. The frequency
        where the ramp will begin.
    :param end_freq: For 'fft_highpass' and 'fft_lowpass'. The frequency
        where the ramp will end.
    '''
    tsd = tsutils.read_iso_ts(input_ts)
    from tstoolbox import filters

    # fft_lowpass, fft_highpass
    if filter_type == 'fft_lowpass':
        return filters.fft_lowpass(tsd, start_freq, end_freq)
    elif filter_type == 'fft_highpass':
        return filters.fft_highpass(tsd, start_freq, end_freq)
    elif filter_type in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        if len(tsd.values) < window_len:
            raise ValueError("Input vector needs to be bigger than window size.")
        if window_len < 3:
            return tsd
        s = pd.np.r_[tsd[window_len - 1:0:-1], tsd, tsd[-1:-window_len:-1]]

        if filter_type == 'flat':  # moving average
            w = pd.np.ones(window_len, 'd')
        else:
            w = eval('pd.np.' + filter_type + '(window_len)')

        y = pd.np.convolve(w / w.sum(), s, mode='valid')
        return y


def zero_crossings(y_axis, window=11):
    """
    Algorithm to find zero crossings. Smoothens the curve and finds the
    zero-crossings by looking for a sign change.


    keyword arguments:
    y_axis -- A list containg the signal over which to find zero-crossings
    window -- the dimension of the smoothing window; should be an odd integer
        (default: 11)

    return -- the index for each zero-crossing
    """
    # smooth the curve
    length = len(y_axis)
    x_axis = np.asarray(range(length), int)

    ymean = y_axis.mean()
    y_axis = y_axis - ymean

    # discard tail of smoothed signal
    y_axis = _smooth(y_axis, window)[:length]
    zero_crossings = np.where(np.diff(np.sign(y_axis)))[0]
    indices = [x_axis[index] for index in zero_crossings]

    # check if zero-crossings are valid
    diff = np.diff(indices)
#    if diff.std() / diff.mean() > 0.2:
#        print diff.std() / diff.mean()
#        print np.diff(indices)
#        raise(ValueError,
#            "False zero-crossings found, indicates problem {0} or {1}".format(
#            "with smoothing window", "problem with offset"))
    # check if any zero crossings were found
    if len(zero_crossings) < 1:
        raise ValueError

    return indices
    # used this to test the fft function's sensitivity to spectral leakage
    # return indices + np.asarray(30 * np.random.randn(len(indices)), int)

# Frequency calculation#############################
#    diff = np.diff(indices)
#    time_p_period = diff.mean()
#
#    if diff.std() / time_p_period > 0.1:
#        raise ValueError,
#            "smoothing window too small, false zero-crossing found"
#
#    #return frequency
#    return 1.0 / time_p_period
##############################################################################

    return tsutils.print_input(print_input, tsd, tmptsd, '_filter')


@baker.command
def read(*filenames):
    '''
    Collect time series from a list of pickle or csv files then print
    in the tstoolbox standard format.

    :param filenames: List of filenames to read time series from.
    '''
    fnames = {}
    for index, filename in enumerate(filenames):
        fname = os.path.basename(os.path.splitext(filename)[0])
        tsd = _sniff_filetype(filename)

        nfname = fname
        if fname in fnames:
            nfname = fname + '_' + str(index)
        fnames[fname] = 1

        if len(filenames) > 1:
            tsd = tsd.rename(columns=lambda x: nfname + '_' + x)

        try:
            result = result.join(tsd)
        except NameError:
            result = tsd

    return tsutils.printiso(result)


@baker.command
def date_slice(start_date=None, end_date=None, input_ts='-'):
    '''
    Prints out data to the screen between start_date and end_date

    :param start_date: The start_date of the series in ISOdatetime format, or
        'None' for beginning.
    :param end_date: The end_date of the series in ISOdatetime format, or
        'None' for end.
    :param input_ts: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    return tsutils.printiso(_date_slice(input_ts=input_ts, start_date=start_date, end_date=end_date))


@baker.command
def describe(input_ts='-'):
    '''
    Prints out statistics for the time-series.

    :param input_ts: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    tsd = tsutils.read_iso_ts(input_ts)
    return tsutils.printiso(tsd.describe())


@baker.command
def peak_detection(method='rel',
                   type='peak',
                   window=24,
                   pad_len=5,
                   points=9,
                   lock_frequency=False,
                   print_input=False,
                   input_ts='-'):
    '''
    Peak and valley detection.

    :param type: 'peak', 'valley', or 'both' to determine what should be
        returned.  Default is 'peak'.
    :param method: 'rel', 'minmax', 'zero_crossing', 'parabola', 'sine'
        methods are available.  The different algorithms have different
        strengths and weaknesses.  The 'rel' algorithm is the default.
    :param window: There will not usually be multiple peaks within the window
        number of values.  The different `method`s use this variable in
        different ways.
        For 'rel' the window keyword specifies how many points on each side
        to require a `comparator`(n,n+x) = True.
        For 'minmax' the window keyword is the distance to look ahead from a
        peak candidate to determine if it is the actual peak
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
        For 'zero_crossing' the window keyword is the dimension of the
        smoothing window; should be an odd integer
    :param points: For 'parabola' and 'sine' methods. How many points around
        the peak should be used during curve fitting, must be odd
        (default: 9)
    :param lock_frequency: For 'sine method only.  Specifies if the
        frequency argument of the model function should be locked to the
        value calculated from the raw peaks or if optimization process may
        tinker with it. (default: False)
    :param print_input: If set to 'True' will include the input columns in
        the output table.  Default is 'False'.
    :param input_ts: Filename with data in 'ISOdate,value' format or '-' for
        stdin.  Default is stdin.
    '''
    # Couldn't get fft method working correctly.  Left pieces in
    # in case want to figure it out in the future.

    if type not in ['peak', 'valley', 'both']:
        raise ValueError(
"""
    The `type` argument must be one of 'peak',
    'valley', or 'both'.  You supplied {0}.
""".format(type))

    if method not in ['rel', 'minmax', 'zero_crossing', 'parabola', 'sine']:
        raise ValueError(
"""
    The `method` argument must be one of 'rel', 'minmax',
    'zero_crossing', 'parabola', or 'sine'.  You supplied {0}.
""".format(method))

    tsd = tsutils.read_iso_ts(input_ts)

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

    if type == 'peak':
        tmptsd = tsd.rename(columns=lambda x: str(x) + '_peak', copy=True)
    if type == 'valley':
        tmptsd = tsd.rename(columns=lambda x: str(x) + '_valley', copy=True)
    if type == 'both':
        tmptsd = tsd.rename(columns=lambda x: str(x) + '_peak', copy=True)
        tmptsd = tmptsd.join(
            tsd.rename(columns=lambda x: str(x) + '_valley', copy=True))

    for c in tmptsd.columns:
        if method in ['fft', 'parabola', 'sine']:
            maxpeak, minpeak = func(tmptsd[c].values, range(len(tmptsd[c])), **kwds)
        else:
            maxpeak, minpeak = func(tmptsd[c].values, **kwds)
        if c[-5:] == '_peak':
            datavals = maxpeak
        if c[-7:] == '_valley':
            datavals = minpeak
        maxx, maxy = list(zip(*datavals))
        hold = tmptsd[c][array(maxx).astype('i')]
        tmptsd[c][:] = nan
        tmptsd[c][array(maxx).astype('i')] = hold

    return tsutils.print_input(print_input, tsd, tmptsd, None)


@baker.command
def convert(factor=1.0, offset=0.0, print_input=False, input_ts='-'):
    '''
    Converts values of a time series by applying a factor and offset.  See the
        'equation' subcommand for a generalized form of this command.

    :param factor: Factor to multiply the time series values.
    :param offset: Offset to add to the time series values.
    :param print_input: If set to 'True' will include the input columns in the
        output table.  Default is 'False'.
    :param input_ts: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    tsd = tsutils.read_iso_ts(input_ts)
    tmptsd = tsd * factor + offset
    return tsutils.print_input(print_input, tsd, tmptsd, '_convert')


@baker.command
def equation(equation, print_input=False, input_ts='-'):
    '''
    Applies <equation> to the time series data.  The <equation> argument is a
        string contained in single quotes with 'x' used as the variable
        representing the input.  For example, '(1 - x)*sin(x)'.

    :param equation: String contained in single quotes that defines the
        equation.  The input variable place holder is 'x'.  Mathematical
        functions in the 'np' (numpy) name space can be used.  For example,
        'x*4 + 2', 'x**2 + cos(x)', and 'tan(x*pi/180)' are all valid
        <equation> strings.  The variable 't' is special representing the time
        at which 'x' occurs.  This means you can so things like 'x[t] +
        max(x[t-1], x[t+1])*0.6' to add to the current value 0.6 times the
        maximum adjacent value.
    :param print_input: If set to 'True' will include the input columns in the
        output table.  Default is 'False'.
    :param input_ts: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    x = tsutils.read_iso_ts(input_ts)
    import re

    # Get rid of spaces
    equation = equation.replace(' ', '')

    tsearch = re.search(r'\[.*?t.*?\]', equation)
    nsearch = re.search(r'x[1-9][0-9]*?', equation)
    # This beasty is so users can use 't' in their equations
    # Indices of 'x' are a function of 't' and can possibly be negative or
    # greater than the length of the DataFrame.
    # Correctly (I think) handles negative indices and indices greater
    # than the length by setting to nan
    # AssertionError happens when index negative.
    # IndexError happens when index is greater than the length of the
    # DataFrame.
    # UGLY!
    if tsearch and nsearch:
        neq = equation.split('[')
        neq = [i.split(']') for i in neq]
        nequation = []
        for i in neq:
            mid = []
            for j in i:
                if 't' in j:
                    mid.append('.values[{0}:{0}+1][0]'.format(j))
                else:
                    mid.append(j)
            nequation.append(mid)
        nequation = [']'.join(i) for i in nequation]
        nequation = '['.join(nequation)
        nequation = re.sub(
            r'x([1-9][0-9]*?)(?!\[)', r'x[x.columns[\1-1]][t]', nequation)
        nequation = re.sub(
            r'x([1-9][0-9]*?)(?=\[)', r'x[x.columns[\1-1]]', nequation)
        nequation = re.sub(
            r'\[\.values', r'.values', nequation)
        nequation = re.sub(
            r'\[0\]\]', r'[0]', nequation)
        y = pd.Series(x[x.columns[0]], index=x.index)
        for t in range(len(x)):
            try:
                y[t] = eval(nequation)
            except (AssertionError, IndexError):
                y[t] = nan
        y = pd.DataFrame(y, columns=['_'])
    elif tsearch:
        y = x.copy()
        nequation = re.sub(
            r'\[(.*?t.*?)\]', r'[col].values[\1:\1+1][0]', equation)
        # Replace 'x' with underlying equation, but not the 'x' in a word, like 'maximum'.
        nequation = re.sub(
            r'(?<![a-zA-Z])x(?![a-zA-Z\[])', r'x[col].values[t:t+1][0]', nequation)
        for col in x.columns:
            for t in range(len(x)):
                try:
                    y[col][t] = eval(nequation)
                except (AssertionError, IndexError):
                    y[col][t] = nan
    elif nsearch:
        nequation = re.sub(
            r'x([1-9][0-9]*?)', r'x[x.columns[\1-1]]', equation)
        y = pd.DataFrame(eval(nequation), columns=['_'])
    else:
        y = eval(equation)
    return tsutils.print_input(print_input, x, y, '_equation')


@baker.command
def pick(columns, input_ts='-'):
    '''
    Will pick a column or list of columns from input.  Start with 1.

    :param columns: Either an integer to collect a single column or a list of
        integers to collect multiple columns.
    :param input_ts: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    tsd = tsutils.read_iso_ts(input_ts)

    columns = columns.split(',')
    columns = [int(i) - 1 for i in columns]

    if len(columns) == 1 and isinstance(columns[0], int):
        return tsutils.printiso(pd.DataFrame(tsd[tsd.columns[columns]]))

    for index, col in enumerate(columns):
        jtsd = pd.DataFrame(tsd[tsd.columns[col]])

        jtsd = jtsd.rename(columns=lambda x: str(x) + '_' + str(index), copy=True)
        try:
            newtsd = newtsd.join(jtsd)
        except NameError:
            newtsd = jtsd
    return tsutils.printiso(newtsd)


@baker.command
def stdtozrxp(rexchange=None, input_ts='-'):
    '''
    Prints out data to the screen in a WISKI ZRXP format.

    :param rexchange: The REXCHANGE ID to be written inro the zrxp header.
    :param input_ts: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    tsd = tsutils.read_iso_ts(input_ts)
    if len(tsd.columns) > 1:
        raise ValueError("""
        The "stdtozrxp" command can only accept a single
        'time-series, instead it is seeing {0}""".format(len(tsd.columns)))
    if rexchange:
        print('#REXCHANGE{0}|*|'.format(rexchange))
    for i in range(len(tsd)):
        print(('{0.year:04d}{0.month:02d}{0.day:02d}{0.hour:02d}'
               '{0.minute:02d}{0.second:02d}, {1}').format(
            tsd.index[i], tsd[tsd.columns[0]][i]))


@baker.command
def tstopickle(filename, input_ts='-'):
    '''
    Pickles the data into a Python pickled file.  Can be brought back into
    Python with 'pickle.load' or 'numpy.load'.  See also 'tstoolbox read'.

    :param filename: The filename to store the pickled data.
    :param input_ts: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    tsd = tsutils.read_iso_ts(input_ts)
    pd.core.common.save(tsd, filename)


@baker.command
def exp_weighted_rolling_window(span=2, statistic='mean', center=False, print_input=False, input_ts='-'):
    '''
    Calculates an exponentially weighted moving window statistic.

    :param span: The number of previous intervals to include in the
        calculation of the statistic.
    :param statistic: 'expw_mean', 'expw_stdev', 'expw_variance'
        'expw_corr', 'expw_cov', defaults to 'expw_mean'.
    :param center: If set to 'True' the calculation will be made for the
        value at the center of the window.  Default is 'False'.
    :param print_input: If set to 'True' will include the input columns in
        the output table.  Default is 'False'.
    :param input_ts: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    tsd = tsutils.read_iso_ts(input_ts)
    if span is None:
        span = len(tsd)
    else:
        span = int(span)
    if statistic == 'expw_mean':
        newts = pd.stats.moments.ewma(tsd, span=span, center=center)
    elif statistic == 'expw_stdev':
        newts = pd.stats.moments.ewmstd(tsd, span=span, center=center)
    elif statistic == 'expw_variance':
        newts = pd.stats.moments.ewmvar(tsd, span=span, center=center)
    elif statistic == 'expw_corr':
        newts = pd.stats.moments.ewmcorr(tsd, span=span, center=center)
    elif statistic == 'expw_cov':
        newts = pd.stats.moments.ewmcov(tsd, span=span, center=center)
    else:
        print('statistic ', statistic, ' is not implemented.')
        sys.exit()
    return tsutils.print_input(print_input, tsd, newts, '_' + statistic)


@baker.command
def accumulate(statistic='sum', print_input=False, input_ts='-'):
    '''
    Calculates accumulating statistics.

    :param statistic: 'sum', 'max', 'min', 'prod', defaults to 'sum'.
    :param print_input: If set to 'True' will include the input columns in
        the output table.  Default is 'False'.
    :param input_ts: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    tsd = tsutils.read_iso_ts(input_ts)
    if statistic == 'sum':
        ntsd = tsd.cumsum()
    elif statistic == 'max':
        ntsd = tsd.cummax()
    elif statistic == 'min':
        ntsd = tsd.cummin()
    elif statistic == 'prod':
        ntsd = tsd.cumprod()
    else:
        raise ValueError('statistic ', statistic, ' is not implemented.')
    return tsutils.print_input(print_input, tsd, ntsd, '_' + statistic)


@baker.command
def rolling_window(span=2, statistic='mean', wintype=None, center=False, print_input=False, input_ts='-'):
    '''
    Calculates a rolling window statistic.

    :param span: The number of previous intervals to include in the
        calculation of the statistic. If `span` is equal to 0 will give an
        expanding rolling window.
    :param statistic: 'mean', 'corr', 'count', 'cov', 'kurtosis', 'median',
        'skew', 'stdev', 'sum', 'variance', used to calculate the
        value, defaults to 'mean'.
    :param wintype: The 'mean' and 'sum' `statistic` calculation can also be
        weighted according to the `wintype` windows.  Some of the following
        windows require additional keywords identified in parenthesis:
        'boxcar', 'triang', 'blackman', 'hamming', 'bartlett', 'parzen',
        'bohman', 'blackmanharris', 'nuttall', 'barthann', 'kaiser' (needs
        beta), 'gaussian' (needs std), 'general_gaussian' (needs power,
        width) 'slepian' (needs width).
    :param center: If set to 'True' the calculation will be made for the
        value at the center of the window.  Default is 'False'.
    :param print_input: If set to 'True' will include the input columns in
        the output table.  Default is 'False'.
    :param input_ts: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    tsd = tsutils.read_iso_ts(input_ts)
    if span is None:
        span = len(tsd)
    else:
        span = int(span)
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
    if wintype in window_list and statistic in ['mean', 'sum']:
        meantest = statistic == 'mean'
        newts = pd.stats.moments.rolling_window(tsd, span, wintype,
                center=center, mean=meantest)
    elif statistic == 'mean':
        if span == 0:
            newts = pd.stats.moments.expanding_mean(tsd, center=center)
        else:
            newts = pd.stats.moments.rolling_mean(tsd, span, center=center)
    elif statistic == 'corr':
        if span == 0:
            newts = pd.stats.moments.expanding_corr(tsd, center=center)
        else:
            newts = pd.stats.moments.rolling_corr(tsd, span, center=center)
    elif statistic == 'cov':
        if span == 0:
            newts = pd.stats.moments.expanding_cov(tsd, center=center)
        else:
            newts = pd.stats.moments.rolling_cov(tsd, span, center=center)
    elif statistic == 'count':
        if span == 0:
            newts = pd.stats.moments.expanding_count(tsd, center=center)
        else:
            newts = pd.stats.moments.rolling_count(tsd, span, center=center)
    elif statistic == 'kurtosis':
        if span == 0:
            newts = pd.stats.moments.expanding_kurt(tsd, center=center)
        else:
            newts = pd.stats.moments.rolling_kurt(tsd, span, center=center)
    elif statistic == 'median':
        if span == 0:
            newts = pd.stats.moments.expanding_median(tsd, center=center)
        else:
            newts = pd.stats.moments.rolling_median(tsd, span, center=center)
    elif statistic == 'skew':
        if span == 0:
            newts = pd.stats.moments.expanding_skew(tsd, center=center)
        else:
            newts = pd.stats.moments.rolling_skew(tsd, span, center=center)
    elif statistic == 'stdev':
        if span == 0:
            newts = pd.stats.moments.expanding_std(tsd, center=center)
        else:
            newts = pd.stats.moments.rolling_std(tsd, span, center=center)
    elif statistic == 'sum':
        if span == 0:
            newts = pd.stats.moments.expanding_sum(tsd, center=center)
        else:
            newts = pd.stats.moments.rolling_sum(tsd, span, center=center)
    elif statistic == 'variance':
        if span == 0:
            newts = pd.stats.moments.expanding_var(tsd, center=center)
        else:
            newts = pd.stats.moments.rolling_var(tsd, span, center=center)
    else:
        raise ValueError('statistic ', statistic, ' is not implemented.')
    return tsutils.print_input(print_input, tsd, newts, '_' + statistic)


@baker.command
def aggregate(statistic='mean',
              agg_interval='daily',
              print_input=False,
              input_ts='-'):
    '''
    Takes a time series and aggregates to specified frequency, outputs to
        'ISO-8601date,value' format.

    :param statistic: 'mean', 'sum', 'std', 'max', 'min', 'median', 'first',
        or 'last' to calculate the aggregation, defaults to 'mean'.
        Can also be a comma separated list of statistic methods.
    :param agg_interval: The 'hourly', 'daily', 'monthly', 'yearly'
        aggregation intervals, defaults to 'daily'.
    :param print_input: If set to 'True' will include the input columns in
        the output table.  Default is 'False'.
    :param input_ts: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    aggd = {'hourly': 'H',
            'daily': 'D',
            'monthly': 'M',
            'yearly': 'A'
            }
    tsd = tsutils.read_iso_ts(input_ts)
    methods = statistic.split(',')
    for method in methods:
        tmptsd = tsd.resample(aggd[agg_interval], how=method)
        try:
            newts = newts.join(tmptsd)
        except NameError:
            newts = tmptsd
    return tsutils.print_input(print_input, tsd, newts, '_' + statistic)


@baker.command
def calculate_fdc(x_plotting_position='norm', input_ts='-'):
    '''
    Returns the frequency distrbution curve.  DOES NOT return a time-series.

    :param x_plotting_position: 'norm' or 'lin'.  'norm' defines a x
        plotting position Defaults to 'norm'.
    :param input_ts: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    from scipy.stats.distributions import norm

    tsd = tsutils.read_iso_ts(input_ts)
    if len(tsd.columns) > 1:
        raise ValueError(
"""
This function currently only works with one time-series at a time.
You gave it {0}""".format(len(tsd.columns)))

    cnt = len(tsd.values)
    a_tmp = 1. / (cnt + 1)
    b_tmp = 1 - a_tmp
    plotpos = ma.empty(len(tsd), dtype=float)
    if x_plotting_position == 'norm':
        plotpos[:cnt] = norm.ppf(linspace(a_tmp, b_tmp, cnt))
        xlabel = norm.cdf(plotpos)
    if x_plotting_position == 'lin':
        plotpos[:cnt] = linspace(a_tmp, b_tmp, cnt)
        xlabel = plotpos
    ydata = ma.sort(tsd[tsd.columns[0]].values, endwith=False)[::-1]
    for xdat, ydat, zdat in zip(plotpos, ydata, xlabel):
        print(xdat, ydat, zdat)


@baker.command
def plot(ofilename='plot.png', xtitle='Time', ytitle='',
         title='', figsize=(10, 6.5), legend=True,
         legend_names=None, input_ts='-'):
    '''
    Time series plot.

    :param ofilename: Output filename for the plot.  Extension defines the
       type, ('.png'). Defaults to 'plot.png'.
    :param xtitle: Title of x-axis, defaults to 'Time'.
    :param ytitle: Title of y-axis, defaults to ''.
    :param title: Title of chart, defaults to ''.
    :param figsize: The (width, height) of plot as inches.  Defaults to
       (10,6.5).
    :param input_ts: Filename with data in 'ISOdate,value' format or '-' for
       stdin.
    '''
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    tsd = tsutils.read_iso_ts(input_ts)
    if legend_names:
        lnames = legend_names.split(',')
        if len(tsd.columns) == len(lnames):
            renamedict = dict(zip(tsd.columns, lnames))
            tsd.rename(columns=renamedict, inplace=True)
        else:
            raise ValueError('''
*
*   For 'legend_names' you must have the same number of comma
*   separated names as columns in the input data.  The input
*   data has {0} where the number of 'legend_names' is {1}.
*
'''.format(len(tsd.columns), len(lnames)))
    plt.figure(figsize=figsize)
    tsd.plot(legend=legend)
    plt.legend(loc='best')
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.title(title)
    plt.savefig(ofilename)


def plotcalibandobs(mwdmpath, mdsn, owdmpath, odsn, ofilename):
    ''' IN DEVELOPMENT Plot model and observed data.
    :param mwdmpath: Path and WDM filename with model data (<64 characters).
    :param mdsn: DSN that contains the model data.
    :param owdmpath: Path and WDM filename with obs data (<64 characters).
    :param mdsn: DSN that contains the observed data.
    :param ofilename: Output filename for the plot.
    '''
    # Create hydrograph plot
    hydroargs = {'color': 'blue',
                 'lw': 0.5,
                 'ls': 'steps-post',
                 }
    hyetoargs = {'c': 'blue',
                 'lw': 0.5,
                 'ls': 'steps-post',
                 }
    fig = cpl.hydrograph(rain['carr'], obs['carr'], figsize=figsize,
                         hydroargs=hydroargs, hyetoargs=hyetoargs)
    # The following just plots 1 dot - I need the line parameters in lin1 to
    # make the legend
    lin1 = fig.hydro.plot(obs['carr'].dates[0:1], obs['carr'][0:1],
                          color='blue', ls='-', lw=0.5, label=line[11])
    rlin1 = fig.hyeto.plot(rain['carr'].dates[0:1], rain['carr'][0:1],
                           color='blue', ls='-', lw=0.5, label=line[9])
    lin2 = fig.hydro.plot(sim['carr'].dates, sim['carr'], color='red', ls=':',
                          lw=1, label='Simulated')
    fig.hyeto.set_ylabel("Total Daily\nRainfall (inches)")
    fig.hyeto.set_ylim(ymax=max_daily_rain, ymin=0)
    fig.hydro.set_ylabel("Average Daily {0} {1}".format(line[14],
                                                        parunitmap[line[14]]))
    if min(obs['arr']) > 0 and line[14] == 'Flow':
        fig.hydro.set_ylim(ymin=0)
    # fig.hydro.set_yscale("symlog", linthreshy=100)
    fig.hydro.set_dlim(plot_start_date, plot_end_date)
    fig.hydro.set_xlabel("Date")
    # fig.legend((lin1, lin2), (line[11], 'Simulated'), (0.6, 0.55))
    make_legend(fig.hydro, (lin1[-1], lin2[-1]))
    make_legend(fig.hyeto, (rlin1[-1],))
    fig.savefig(newbasename + "_daily_hydrograph.png")
    fig.savefig(rnewbasename + "_daily_hydrograph.png")


def main():
    baker.run()
