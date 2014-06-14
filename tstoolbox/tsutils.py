'''
A collection of functions used by tstoolbox, wdmtoolbox, ...etc.
'''

from __future__ import print_function
from __future__ import division

import os

import pandas as pd
import numpy as np


def date_slice(input_tsd, start_date=None, end_date=None):
    '''
    Private function to slice time series.
    '''

    if input_tsd.index.is_all_dates:
        accdate = []
        for testdate,alpha_omega in [(start_date, 0), (end_date, -1)]:
            if testdate is None:
                tdate = input_tsd.index[alpha_omega]
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


def asbestfreq(data):
    ''' This uses PANDAS .asfreq.  Basically, how low
    can you go and maintain the same number of values.
    '''

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
        # Following test would work if NO missing data.
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
            for tstep in range(int(minterval)//int(finterval) + 1, 0, -1):
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
                        float_format=float_format)
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
    import sys
    sys.tracebacklimit = 1000
    try:
        if tsd.index.is_all_dates:
            tsd.index.name = 'Datetime'
            tsd.to_csv(sys.stdout, float_format=float_format,
                       date_format=date_format, sep=sep)
        else:
            tsd.index.name = 'UniqueID'
            tsd.to_csv(sys.stdout, float_format=float_format,
                       date_format=date_format, sep=sep)
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


def read_iso_ts(indat, dense=True, parse_dates=True):
    '''
    Reads the format printed by 'print_iso' and maybe other formats.
    '''
    import csv
    from pandas.compat import StringIO

    import baker

    index_col = 0
    if parse_dates is False:
        index_col = False

    # Would want this to be more generic...
    na_values = []
    for spc in range(20)[1:]:
        spcs = ' '*spc
        na_values.append(spcs)
        na_values.append(spcs + 'nan')

    fp = None

    # Handle Series by converting to DataFrame
    if isinstance(indat, pd.Series):
        indat = pd.DataFrame(indat)

    if isinstance(indat, pd.DataFrame):
        if indat.index.is_all_dates:
            indat.index.name = 'Datetime'
            if dense:
                return asbestfreq(indat)[0]
            else:
                return indat
        else:
            indat.index.name = 'UniqueID'
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
            fpi = baker.openinput(indat)
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
                fpi = open(indat)
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
            dialect = csv.Sniffer().sniff(readsome,
                                          delimiters=', \t:|')
            has_header = csv.Sniffer().has_header(readsome)
            fpi.seek(0)
        except:
            pass

        if has_header:
            result = pd.io.parsers.read_table(fpi, header=0,
                                              dialect=dialect,
                                              index_col=index_col,
                                              parse_dates=True,
                                              skipinitialspace=True)
            result.columns = [i.strip() for i in result.columns]
        else:
            result = pd.io.parsers.read_table(fpi, header=None,
                                              dialect=dialect,
                                              index_col=0,
                                              parse_dates=True)
            fname = os.path.splitext(fpi.name)
            if len(result.columns) == 1:
                result.columns = [fname[0]]
            else:
                result.columns = ['{0}_{1}'.format(fname[0], i)
                                  for i in result.columns]

    if result.index.is_all_dates is True:
        result.index.name = 'Datetime'

        if dense:
            try:
                return asbestfreq(result)[0]
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
