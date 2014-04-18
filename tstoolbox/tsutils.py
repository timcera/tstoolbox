
from __future__ import print_function
from __future__ import division

import os

import pandas as pd
import numpy as np


def date_slice(input_tsd, start_date=None, end_date=None):
    '''
    Private function to slice time series.
    '''
    if start_date is None:
        sdate = input_tsd.index[0]
    else:
        sdate = pd.Timestamp(start_date)
    if end_date is None:
        edate = input_tsd.index[-1]
    else:
        edate = pd.Timestamp(end_date)
    ltsd = len(input_tsd.columns)
    if sdate < input_tsd.index[0]:
        before = pd.DataFrame([[pd.np.nan]*ltsd], index=[sdate],
                columns=input_tsd.columns)
        input_tsd = before.append(input_tsd)
    if edate > input_tsd.index[-1]:
        after = pd.DataFrame([[pd.np.nan]*ltsd], index=[edate],
                columns=input_tsd.columns)
        input_tsd = input_tsd.append(after)
    return input_tsd[sdate:edate]


def asbestfreq(data):
    # This uses PANDAS .asfreq.  Basically, how low
    # can you go and maintain the same number of values.

    # Since pandas doesn't set data.index.freq and data.index.freqstr when
    # using .asfreq, this function returns that PANDAS time offset alias code
    # also.  Not ideal at all.

    # This gets most of the frequencies...
    try:
        return data.asfreq(data.index.inferred_freq), data.index.inferred_freq
    except ValueError:
        pass

    pandacodes = ['A', 'AS', 'BA', 'BAS',    # Annual
                  'Q', 'QS', 'BQ', 'BQS',    # Quarterly
                  'M', 'MS', 'BM', 'BMS',    # Monthly
                  'W',                       # Weekly
                  'D', 'B',                  # Daily
                  'H', 'T', 'S', 'L', 'U']   # Intra-daily

    # This first loop gets the basic offset alias
    cnt = data.count()
    for pandacode in pandacodes:
        tstfreq = data.asfreq('{0}'.format(pandacode))
        if np.all(tstfreq.count() == cnt):
            break

    # Now need to find the tstep, for example bi-weekly = '2W'

    # Start with the minimum interval after dropping all NaNs.
    interval = np.unique(tstfreq.dropna(how='all').index.values[1:] -
                         tstfreq.dropna(how='all').index.values[:-1])
    minterval = np.min(interval)

    codemap = {'W': 604800000000000,
               'D': 86400000000000,
               'H': 3600000000000,
               'T': 60000000000,
               'S': 1000000000,
               'L': 1000000,
               'U': 1000,
               }

    finterval = codemap.setdefault(pandacode, None)

    tstep = 1
    if finterval == minterval:
        return tstfreq, pandacode
    elif finterval is not None:
        try:
            for tstep in range(int(minterval)//int(finterval) + 1, 1, -1):
                tstfreq = data.asfreq('{0}{1}'.format(tstep, pandacode))
                if np.all(tstfreq.count() == cnt):
                    break
        except AttributeError:
            # Maybe figure out how to handle inconsistent intervals, like 'M'.
            # That would go here....
            pass

    return tstfreq, '{0}{1}'.format(tstep, pandacode)


# Utility
def print_input(iftrue, intds, output, suffix,
                date_format=None, sep=',',
                float_format='%g'):
    if suffix:
        output.rename(columns=lambda xloc: xloc + suffix, inplace=True)
    if iftrue:
        return printiso(intds.join(output, how='outer'),
                        date_format=date_format,
                        sep=sep, float_format=float_format)
    else:
        return printiso(output, date_format=date_format, sep=sep,
                        float_format=float_format)


def _apply_across_columns(func, xtsd, **kwds):
    for col in xtsd.columns:
        xtsd[col] = func(xtsd[col], **kwds)
    return xtsd


def _printiso(tsd, date_format=None, sep=',',
              float_format='%g'):
    ''' Separate so can use in tests.
    '''
    tsd.index.name = 'Datetime'
    import sys
    sys.tracebacklimit = 1000
    try:
        if tsd.index.is_all_dates:
            tsd.index.name = 'Datetime'
            tsd.to_csv(sys.stdout, float_format=float_format,
                       date_format=date_format, sep=sep)
        else:
            print(tsd)
    except IOError:
        return


def printiso(tsd, sparse=False, date_format=None,
             sep=',', float_format='%g'):
    '''
    Default output format for tstoolbox, wdmtoolbox, swmmtoolbox,
    and hspfbintoolbox.
    '''
    import sys
    try:
        oldtracebacklimit = sys.tracebacklimit
    except AttributeError:
        oldtracebacklimit = 1000
    sys.tracebacklimit = 1000
    import traceback
    import os.path
    baker_cli = False
    for i in traceback.extract_stack():
        if os.path.basename(i[0]) == 'baker.py':
            baker_cli = True
            break
    sys.tracebacklimit = oldtracebacklimit
    tsd.index.name = 'Datetime'
    if baker_cli:
        _printiso(tsd, float_format=float_format,
                  date_format=date_format, sep=sep)
    else:
        return tsd


def read_iso_ts(indat, dense=True):
    '''
    Reads the format printed by 'print_iso'.
    '''
    import csv
    from pandas.compat import StringIO, u

    import baker

    fp = None

    # Handle Series by converting to DataFrame
    if isinstance(indat, pd.Series):
        indat = pd.DataFrame(indat)

    if isinstance(indat, pd.DataFrame):
        indat.index.name = 'Datetime'
        if dense:
            return asbestfreq(indat)[0]
        else:
            return indat

    if isinstance(indat, str) or isinstance(indat, bytes):
        try:
            indat = str(indat, encoding='utf-8')
        except:
            pass
        if indat == '-':
            # format must be the tstoolbox standard
            has_header = True
            dialect = csv.excel
            fp = baker.openinput(indat)
        elif '\n' in indat or '\r' in indat:
            # a string
            fp = StringIO(indat)
        elif os.path.exists(indat):
            # Is it a pickled file?
            try:
                result = pd.io.pickle.read_pickle(indat)
                fp = False
            except:
                # Maybe a CSV file?
                fp = open(indat)
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

    if fp:
        try:
            fp.seek(0)
            readsome = fp.read(2048)
            dialect = csv.Sniffer().sniff(readsome,
                                          delimiters=', \t:|')
            has_header = csv.Sniffer().has_header(readsome)
            fp.seek(0)
        except:
            pass

        if has_header:
            result = pd.io.parsers.read_table(fp, header=0,
                                              dialect=dialect,
                                              index_col=0,
                                              parse_dates=True)
            result.columns = [i.strip() for i in result.columns]
        else:
            result = pd.io.parsers.read_table(fp, header=None,
                                              dialect=dialect,
                                              index_col=0,
                                              parse_dates=True)
            fname, ext = os.path.splitext(fp.name)
            if len(result.columns) == 1:
                result.columns = [fname]
            else:
                result.columns = ['{0}_{1}'.format(fname, i)
                                  for i in result.columns]

    result.index.name = 'Datetime'

    if dense:
        try:
            return asbestfreq(result)[0]
        except ValueError:
            return result
    else:
        return result


def read_excel_csv(fp, header=None):
    if header is not None:
        header = int(header)
    tsdata = pd.read_table(fp, header=header, sep=',', parse_dates=[0],
                           index_col=[0])
    return tsdata
