'''
A collection of functions used by tstoolbox, wdmtoolbox, ...etc.
'''

from __future__ import print_function
from __future__ import division

import os
import sys
import gzip
import bz2
try:
    from fractions import gcd
except ImportError:
    from math import gcd
try:
    from functools import reduce
except ImportError:
    pass

import pandas as pd
import numpy as np


def common_kwds(input_tsd,
                start_date=None,
                end_date=None,
                pick=None,
                force_freq=None,
                groupby=None,
                dropna='no'):
    '''Collected all common_kwds across sub-commands into this single function.
    '''
    ntsd = input_tsd
    if pick is not None:
        ntsd = _pick(ntsd, pick)
    if start_date is not None or end_date is not None:
        ntsd = _date_slice(ntsd, start_date=start_date, end_date=end_date)
    if force_freq is not None:
        ntsd = asbestfreq(ntsd, force_freq=force_freq)
    if ntsd.index.is_all_dates:
        ntsd.index.name = 'Datetime'
    if groupby is not None:
        if groupby == 'months_across_years':
            return ntsd.groupby(lambda x: x.month)
        else:
            return ntsd.groupby(pd.TimeGrouper(groupby))
    if dropna not in ['any', 'all', 'no']:
        raise ValueError('''
*
*   The "dropna" option must be "any", "all" or "no", not "{0}".
*
'''.format(dropna))
    else:
        if dropna in ['any', 'all']:
            ntsd = ntsd.dropna(axis='index', how=dropna)
    return ntsd


def _pick(tsd, columns):
    columns = columns.split(',')
    ncolumns = []

    for i in columns:
        if i in tsd.columns:
            # if using column names
            ncolumns.append(tsd.columns.tolist().index(i))
            continue
        elif i == tsd.index.name:
            # if wanting the index
            # making it -1 that will be evaluated later...
            ncolumns.append(-1)
            continue
        else:
            # if using column numbers
            try:
                target_col = int(i) - 1
            except:
                raise ValueError('''
*
*   The name {0} isn't in the list of column names
*   {1}.
*
'''.format(i, tsd.columns))
            if target_col < 0:
                raise ValueError('''
*
*   The request column index {0} must be greater than or equal to 0.
*   First data column is index 1, index is column 0.
*
'''.format(i))
            if target_col > len(tsd.columns):
                raise ValueError('''
*
*   The request column index {0} must be less than the
*   number of columns {1}.
*
'''.format(i, len(tsd.columns)))

            # columns names or numbers or index organized into
            # numbers in ncolumns
            ncolumns.append(target_col)

    if len(ncolumns) == 1 and ncolumns[0] != -1:
        return pd.DataFrame(tsd[tsd.columns[ncolumns]])

    newtsd = pd.DataFrame()
    for index, col in enumerate(ncolumns):
        if col == -1:
            # Use the -1 marker to indicate index
            jtsd = pd.DataFrame(tsd.index)
        else:
            jtsd = pd.DataFrame(tsd[tsd.columns[col]])

        newtsd = newtsd.join(jtsd,
                             lsuffix='_{0}'.format(index), how='outer')
    return newtsd


def date_slice(input_tsd, start_date=None, end_date=None):
    '''
    This is here for a while until I fix my other toolboxes to
    use common_kwds instead.
    '''
    return _date_slice(input_tsd, start_date=start_date, end_date=end_date)


def _date_slice(input_tsd, start_date=None, end_date=None):
    '''
    Private function to slice time series.
    '''

    if input_tsd.index.is_all_dates:
        accdate = []
        for testdate in [start_date, end_date]:
            if testdate is None:
                tdate = None
            else:
                tdate = pd.Timestamp(testdate)
                # Is this comparison cheaper than the .join?
                if not pd.np.any(input_tsd.index == tdate):
                    # Create a dummy column at the date I want, then delete
                    # Not the best, but...
                    row = pd.DataFrame([pd.np.nan], index=[tdate])
                    row.columns = ['deleteme']
                    input_tsd = input_tsd.join(row, how='outer')
                    input_tsd.drop('deleteme', inplace=True, axis=1)
            accdate.append(tdate)

        return input_tsd[slice(*accdate)]
    else:
        return input_tsd

_annuals = {
    0: 'DEC',
    1: 'JAN',
    2: 'FEB',
    3: 'MAR',
    4: 'APR',
    5: 'MAY',
    6: 'JUN',
    7: 'JUL',
    8: 'AUG',
    9: 'SEP',
    10: 'OCT',
    11: 'NOV',
    12: 'DEC',
    }

_weeklies = {
    0: 'MON',
    1: 'TUE',
    2: 'WED',
    3: 'THU',
    4: 'FRI',
    5: 'SAT',
    6: 'SUN',
    }


def asbestfreq(data, force_freq=None):
    '''
    This uses several techniques.
    1. If force_freq is set use .asfreq.
    2. If data.index.freq is not None, just return.
    3. If data.index.inferred_freq is set use .asfreq.
    4. Use pd.infer_freq - fails if any missing
    5. Use .is_* functions to establish A, AS, A-*, AS-*, Q, QS, M, MS
    6. Use minimum interval to establish the fixed time periods up to weekly
    7. Gives up returning None for PANDAS offset string
    '''
    if force_freq is not None:
        return data.asfreq(force_freq)

    if data.index.freq is not None:
        return data

    # Since pandas doesn't set data.index.freq and data.index.freqstr when
    # using .asfreq, this function returns that PANDAS time offset alias code
    # also.  Not ideal at all.

    # This gets most of the frequencies...
    try:
        return data.asfreq(data.index.inferred_freq)
    except ValueError:
        pass

    # pd.infer_freq would fail if given a large dataset
    if len(data.index) > 100:
        slic = slice(None, 99)
    else:
        slic = slice(None, None)
    infer_freq = pd.infer_freq(data.index[slic])
    if infer_freq is not None:
        return data.asfreq(infer_freq)

    # At this point pd.infer_freq failed probably because of missing values.
    # The following algorithm would not capture things like BQ, BQS
    # ...etc.
    if np.alltrue(data.index.is_year_end):
        infer_freq = 'A'
    elif np.alltrue(data.index.is_year_start):
        infer_freq = 'AS'
    elif np.alltrue(data.index.is_quarter_end):
        infer_freq = 'Q'
    elif np.alltrue(data.index.is_quarter_start):
        infer_freq = 'QS'
    elif np.alltrue(data.index.is_month_end):
        if np.all(data.index.month == data.index[0].month):
            # Actually yearly with different ends
            infer_freq = 'A-{0}'.format(_annuals[data.index[0].month])
        else:
            infer_freq = 'M'
    elif np.alltrue(data.index.is_month_start):
        if np.all(data.index.month == data.index[0].month):
            # Actually yearly with different start
            infer_freq = 'A-{0}'.format(_annuals[data.index[0].month] - 1)
        else:
            infer_freq = 'MS'

    if infer_freq is not None:
        return data.asfreq(infer_freq)

    # Use the minimum of the intervals to test a new interval.
    # Should work for fixed intervals.
    ndiff = sorted(set(np.diff(data.index.values.astype(int))))
    mininterval = int(np.min(ndiff))
    if mininterval <= 0:
        raise ValueError
    if len(ndiff) == 1:
        ngcd = ndiff[0]
    else:
        ngcd = reduce(gcd, ndiff)
    if ngcd < 1000:
        infer_freq = '{0}N'.format(ngcd)
    elif ngcd < 1000000:
        infer_freq = '{0}U'.format(ngcd//1000)
    elif ngcd < 1000000000:
        infer_freq = '{0}L'.format(ngcd//1000000)
    elif ngcd < 60000000000:
        infer_freq = '{0}S'.format(ngcd//1000000000)
    elif ngcd < 3600000000000:
        infer_freq = '{0}T'.format(ngcd//60000000000)
    elif ngcd < 86400000000000:
        infer_freq = '{0}H'.format(ngcd//3600000000000)
    elif ngcd < 604800000000000:
        infer_freq = '{0}D'.format(ngcd//86400000000000)
    elif ngcd < 2419200000000000:
        infer_freq = '{0}W'.format(ngcd//604800000000000)
        if np.all(data.index.dayofweek == data.index[0].dayofweek):
            infer_freq = infer_freq + '-{0}'.format(
                _weeklies[data.index[0].dayofweek])
        else:
            infer_freq = 'D'

    if infer_freq is not None:
        return data.asfreq(infer_freq)

    # Give up
    return data


# Utility
def print_input(iftrue, intds, output, suffix,
                date_format=None, sep=',',
                float_format='%g',
                force_print_index=False):
    ''' Used when wanting to print the input time series also.
    '''
    if suffix:
        output.rename(columns=lambda xloc: xloc + suffix, inplace=True)
    if iftrue:
        return printiso(intds.join(output,
                                   lsuffix='_1',
                                   rsuffix='_2',
                                   how='outer'),
                        date_format=date_format,
                        sep=sep,
                        float_format=float_format,
                        force_print_index=force_print_index)
    else:
        return printiso(output, date_format=date_format, sep=sep,
                        float_format=float_format,
                        force_print_index=force_print_index)


def _apply_across_columns(func, xtsd, **kwds):
    ''' Apply a function to each column in turn.
    '''
    for col in xtsd.columns:
        xtsd[col] = func(xtsd[col], **kwds)
    return xtsd


def _printiso(tsd, date_format=None, sep=',',
              float_format='%g', force_print_index=False):
    ''' Separate so can use in tests.
    '''
    sys.tracebacklimit = 1000

    print_index = True
    if tsd.index.is_all_dates:
        if tsd.index.name is None:
            tsd.index.name = 'Datetime'
        # Someone made the decision about the name
        # This is how I include time zone info by tacking on to the
        # index.name.
        elif 'datetime' not in tsd.index.name.lower():
            tsd.index.name = 'Datetime'
    else:
        # This might be overkill, but tstoolbox is for time-series.
        # Revisit if necessary.
        print_index = False

    if tsd.index.name == 'UniqueID':
        print_index = False

    if force_print_index is True:
        print_index = True

    try:
        tsd.to_csv(sys.stdout, float_format=float_format,
                   date_format=date_format, sep=sep, index=print_index)
    except IOError:
        return


def test_cli():
    ''' The structure to test the cli.
    '''
    import traceback
    try:
        oldtracebacklimit = sys.tracebacklimit
    except AttributeError:
        oldtracebacklimit = 1000
    sys.tracebacklimit = 1000
    cli = False
    for i in traceback.extract_stack():
        if os.path.sep + 'mando' + os.path.sep in i[0] or 'baker' in i[0]:
            cli = True
            break
    sys.tracebacklimit = oldtracebacklimit
    return cli


def printiso(tsd, date_format=None,
             sep=',', float_format='%g', force_print_index=False):
    '''
    Default output format for tstoolbox, wdmtoolbox, swmmtoolbox,
    and hspfbintoolbox.
    '''

    # Not perfectly true, but likely will use force_print_index for indices
    # that are not time stamps.
    if force_print_index is True:
        if not tsd.index.name:
            tsd.index.name = 'UniqueID'
    else:
        tsd.index.name = 'Datetime'

    if test_cli():
        _printiso(tsd, float_format=float_format,
                  date_format=date_format, sep=sep,
                  force_print_index=force_print_index)
    else:
        return tsd


def openinput(filein):
    """
    Opens the given input file. It can decode various formats too, such as
    gzip and bz2.
    """
    if filein == '-':
        return sys.stdin
    ext = os.path.splitext(filein)[1]
    if ext in ['.gz', '.GZ']:
        return gzip.open(filein, 'rb')
    if ext in ['.bz', '.bz2']:
        return bz2.BZ2File(filein, 'rb')
    return open(filein, 'rb')


def read_iso_ts(indat,
                dropna='no',
                parse_dates=True,
                extended_columns=False,
                force_freq=None):
    '''
    Reads the format printed by 'print_iso' and maybe other formats.
    '''
    import csv
    from pandas.compat import StringIO

    if force_freq is not None:
        # force_freq implies a dense series
        dropna = 'no'

    index_col = 0
    if parse_dates is False:
        index_col = False

    # Would want this to be more generic...
    na_values = []
    for spc in range(20)[1:]:
        spcs = ' '*spc
        na_values.append(spcs)
        na_values.append(spcs + 'nan')

    fpi = None

    # Handle Series by converting to DataFrame
    if isinstance(indat, pd.Series):
        indat = pd.DataFrame(indat)

    if isinstance(indat, pd.DataFrame):
        if indat.index.is_all_dates:
            indat.index.name = 'Datetime'
            if dropna == 'no':
                return asbestfreq(indat, force_freq=force_freq)
            else:
                return indat
        else:
            indat.index.name = 'UniqueID'
            return indat

    has_header = False
    dialect = csv.excel
    if isinstance(indat, str) or isinstance(indat, bytes):
        try:
            indat = str(indat, encoding='utf-8')
        except:
            pass
        if indat == '-':
            # if from stdin format must be the tstoolbox standard
            has_header = True
            fpi = openinput(indat)
        elif '\n' in indat or '\r' in indat:
            # a string
            fpi = StringIO(indat)
        elif os.path.exists(indat):
            # Is it a pickled file?
            try:
                result = pd.io.pickle.read_pickle(indat)
                fpi = False
            except:
                # Maybe a CSV file?
                fpi = openinput(indat)
        else:
            raise ValueError('''
*
*   File {0} doesn't exist.
*
'''.format(indat))
    else:
        raise ValueError('''
*
*   Can't figure out what was passed to read_iso_ts.
*
''')

    if fpi:
        try:
            fpi.seek(0)
            readsome = fpi.read(2048)
            fpi.seek(0)
            dialect = csv.Sniffer().sniff(readsome,
                                          delimiters=', \t:|')
            has_header = csv.Sniffer().has_header(readsome)
        except:
            # This is an assumption.
            has_header = True

        if extended_columns is True:
            fname = os.path.splitext(os.path.basename(fpi.name))[0]
            fstr = '{0}.{1}'
        else:
            fname = ''
            fstr = '{1}'
        if fname == '<stdin>':
            fname = '_'
        if has_header:
            result = pd.io.parsers.read_table(fpi, header=0,
                                              dialect=dialect,
                                              index_col=index_col,
                                              parse_dates=True,
                                              skipinitialspace=True)
            result.columns = [fstr.format(fname, i.strip())
                              for i in result.columns]
        else:
            result = pd.io.parsers.read_table(fpi, header=None,
                                              dialect=dialect,
                                              index_col=0,
                                              parse_dates=True,
                                              skipinitialspace=True)
            if len(result.columns) == 1:
                result.columns = [fname]
            else:
                result.columns = [fstr.format(fname, i.strip())
                                  for i in result.columns]

    if result.index.is_all_dates is True:
        result.index.name = 'Datetime'

        if dropna == 'no':
            try:
                return asbestfreq(result, force_freq=force_freq)
            except ValueError:
                return result
    else:
        if result.index.name != 'UniqueID':
            result.reset_index(level=0, inplace=True)
        result.index.name = 'UniqueID'
    return result


def read_excel_csv(fpi, header=None):
    ''' Read Excel formatted CSV file.
    '''
    if header is not None:
        header = int(header)
    tsdata = pd.read_table(fpi, header=header, sep=',', parse_dates=[0],
                           index_col=[0])
    return tsdata
