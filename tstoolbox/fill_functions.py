#!/sjr/beodata/local/python_linux/bin/python

from __future__ import print_function

import pandas as pd
from numpy import *
from scipy.interpolate import interp1d
import baker

from tstoolbox import tsutils

_offset_aliases = {
    86400000000000:    'D',
    604800000000000:   'W',
    2419200000000000:  'M',
    2505600000000000:  'M',
    2592000000000000:  'M',
    2678400000000000:  'M',
    31536000000000000: 'A',
    31622400000000000: 'A',
    3600000000000:     'H',
    60000000000:       'M',
    1000000000:        'T',
    1000000:           'L',
    1000:              'U',
    }


@baker.command
def fill(method='ffill', interval='guess', print_input=False,  input_ts='-'):
    '''
    Fills missing values (NaN) with different methods.  Missing values can
        occur because of NaN, or because the time series is sparse.  The
        'interval' option can insert NaNs to create a dense time series.
    :param method: String contained in single quotes or a number that
        defines the method to use for filling.
        'ffill': assigns NaN values to the last good value
        'bfill': assigns NaN values to the next good value
        number: fills all NaN to this number
        'interpolate': will linearly interpolate missing values
        'spline': spline interpolation
    :param interval: Will try to insert missing intervals.  Can give any
        of the pandas offset aliases, 'guess' (to try and figure the
        interval), or None to not insert missing intervals.
    :param print_input: If set to 'True' will include the input columns in the
        output table.  Default is 'False'.
    :param input_ts: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    tsd = tsutils.read_iso_ts(input_ts)
    ntsd = tsd.copy()
    if interval is not None:
        if interval == 'guess':
            interval = pd.np.min(tsd.index.values[1:] - tsd.index.values[:-1])
            try:
                interval = _offset_aliases[interval]
            except KeyError:
                raise ValueError("""
Can't guess interval, you must supply desired interval with
'--interval=' argument.""")
            if interval == 'M':
                # Need to determine whether 'M' or 'MS'
                dr = pd.date_range(tsd.index[0], periods=len(tsd), freq='M')
                if tsd.reindex(dr).count() != tsd.count():
                    interval = 'MS'
        ntsd = tsd.asfreq(interval)
    predf = pd.DataFrame(ntsd.mean().values,
                         index=[ntsd.index[0] - pd.offsets.Hour()])
    predf.columns = ntsd.columns
    postf = pd.DataFrame(ntsd.mean().values,
                         index=[ntsd.index[-1] + pd.offsets.Hour()])
    postf.columns = ntsd.columns
    ntsd = pd.concat([predf, ntsd, postf])
    if method in ['ffill', 'bfill']:
        ntsd = ntsd.fillna(method=method)
    elif method in ['linear']:
        ntsd = ntsd.apply(pd.Series.interpolate, method='values')
    elif method in ['nearest', 'zero', 'slinear', 'quadratic', 'cubic']:
        for c in ntsd.columns:
            df2 = ntsd[c].dropna()
            f = interp1d(df2.index.values.astype('d'), df2.values, kind=method)
            slices = pd.isnull(ntsd[c])
            ntsd[c][slices] = f(ntsd[c][slices].index.values.astype('d'))
    elif method in ['mean']:
        ntsd = ntsd.fillna(ntsd.mean())
    elif method in ['median']:
        ntsd = ntsd.fillna(ntsd.median())
    elif method in ['max']:
        ntsd = ntsd.fillna(ntsd.max())
    elif method in ['min']:
        ntsd = ntsd.fillna(ntsd.min())
    else:
        try:
            ntsd = ntsd.fillna(value=float(method))
        except ValueError:
            pass
    ntsd = ntsd.iloc[1:-1]
    tsutils.print_input(print_input, tsd, ntsd, '_fill')
