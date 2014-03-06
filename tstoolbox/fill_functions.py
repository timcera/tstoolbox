#!/sjr/beodata/local/python_linux/bin/python

from __future__ import print_function
from __future__ import absolute_import

import pandas as pd
from numpy import *
import baker

from . import tsutils


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
        'linear': will linearly interpolate missing values
        'spline': spline interpolation
        'nearest': nearest good value
        'zero':
        'slinear':
        'quadratic':
        'cubic':
        'mean': fill with mean
        'median': fill with median
        'max': fill with maximum
        'min': fill with minimum
        If a number will fill with that number.
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
    ntsd = tsutils.asbestfreq(ntsd)[0]
    offset = ntsd.index[1] - ntsd.index[0]
    predf = pd.DataFrame(dict(zip(tsd.columns, tsd.mean().values)),
                         index=[tsd.index[0] - offset])
    postf = pd.DataFrame(dict(zip(tsd.columns, tsd.mean().values)),
                         index=[tsd.index[-1] + offset])
    ntsd = pd.concat([predf, ntsd, postf])
    if method in ['ffill', 'bfill']:
        ntsd = ntsd.fillna(method=method)
    elif method in ['linear']:
        ntsd = ntsd.apply(pd.Series.interpolate, method='values')
    elif method in ['nearest', 'zero', 'slinear', 'quadratic', 'cubic']:
        from scipy.interpolate import interp1d
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
    tsd.index.name = 'Datetime'
    ntsd.index.name = 'Datetime'
    return tsutils.print_input(print_input, tsd, ntsd, '_fill')


def fill_from_others(method='best',
                     maximum_lag=24,
                     interval='guess',
                     print_input=False,
                     input_ts='-'):
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
    ntsd = tsutils.asbestfreq(ntsd)[0]
