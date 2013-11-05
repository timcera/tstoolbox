
from __future__ import print_function


import pandas as pd
from dateutil.parser import parse
import numpy as np


def _isfinite(testval):
    '''
    Private utility for 'printiso' function.
    Just returns a blank in place of 'nan' so that other applications see just
    a missing value.
    '''
    try:
        torf = np.isfinite(float(testval))
        if torf:
            return str(testval)
        else:
            return ' '
    except (TypeError, ValueError):
        return ' '


def guess_freq(data):
    # Another way to do this is to abuse PANDAS .asfreq.  Basically, how low
    # can you go and maintain the same number of values.
    mapcode = {365*86400: 6,
               366*86400: 6,
               31*86400:  5,
               30*86400:  5,
               29*86400:  5,
               28*86400:  5,
               86400:     4,
               3600:      3,
               60:        2,
               1:         1
               }

    pndcode = {365*86400: 'A',
               366*86400: 'A',
               31*86400:  'M',
               30*86400:  'M',
               29*86400:  'M',
               28*86400:  'M',
               86400:     'D',
               3600:      'H',
               60:        'T',
               1:         'S'
               }

    import itertools
    import fractions

    if data.index.freq is not None:
        return mapcode[data.index.freq.nanos/1000000000], data.index.freqstr

    interval = np.unique(data.index.values[1:] -
                         data.index.values[:-1])/1000000000
    interval = interval.tolist()

    # If there are more than one interval lets see if the are caused by
    # missing values.  Say there is at least one 2 hour interval and at
    # least one 1 hour interval, this should correctly say the interval
    # is one hour.
    ninterval = set()
    if len(interval) > 1 and np.all(np.array(interval) < 2419200):
        for aval, bval in itertools.combinations(interval, 2):
            ninterval.add(fractions.gcd(int(aval), int(bval)))
        interval = list(ninterval)
        interval.sort()

    # If len of intervai is STILL > than 1, must be large time spans between
    # data.  Lets try to figure out the largest common interval that will
    # evenly fit in the observation intervals.
    if len(interval) > 1:
        accumulate_freq = []
        for inter in interval:
            for seconds in sorted(mapcode, reverse=True):
                if seconds > inter:
                    continue
                if inter % seconds == 0:
                    accumulate_freq.append(seconds)
                    break

        accumulate_freq.sort()
        finterval = accumulate_freq[0]
    else:
        finterval = interval[0]

    pandacode = pndcode[finterval]
    if pandacode == 'M':
        if data.index[0].day == 1:
            pandacode = 'MS'
    elif pandacode == 'A':
        if data.index[0].month == 1:
            pandacode = 'AS'

    return mapcode[finterval], pandacode


# Utility
def print_input(iftrue, input, output, suffix):
    if suffix:
        output.rename(columns=lambda xloc: xloc + suffix, inplace=True)
    if iftrue:
        return printiso(input.join(output, how='outer'))
    else:
        return printiso(output)


def _printiso(tsd):
    ''' Separate so can use in tests.
    '''
    import sys
    sys.tracebacklimit = 1000
    fp = open('tmplog.txt', 'w')
    try:
        if tsd.index.is_all_dates:
            # Header
            print('Datetime,', ', '.join(str(i) for i in tsd.columns))

            # Data
            for i in range(len(tsd)):
                fp.write(str(tsd.values[i]) + '\n')
                print(tsd.index[i], ', ', ', '.join(
                    _isfinite(j) for j in tsd.values[i]))
        else:
            print(tsd)
    except IOError:
        return


def printiso(tsd, sparse=False):
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
    if baker_cli:
        _printiso(tsd)
    else:
        return tsd


def read_iso_ts(indat, dense=True):
    '''
    Reads the format printed by 'print_iso'.
    '''
    import baker

    if isinstance(indat, pd.DataFrame):
        if dense:
            return indat.asfreq(guess_freq(indat)[1])
        else:
            return indat

    fp = baker.openinput(indat)

    header = fp.readline()
    try:
        header = str(header, encoding='utf8')
    except TypeError:
        header = str(header)
    header = header.split(',')
    header = [i.strip() for i in header]
    dates = []
    values = {}
    for line in fp:
        try:
            # Python 3
            nline = str(line, encoding='utf8')
        except TypeError:
            # Python 2
            nline = str(line)
        words = nline.split(',')

        if len(words) != len(header):
            raise ValueError('''
*
*  The number of columns in record:
*   {0}
*  does not match the number of column names in the header:
*   {1}
*
'''.format(words, header))

        dates.append(parse(words[0]))
        for index, col in enumerate(header[1:]):
            try:
                values.setdefault(col, []).append(float(words[1 + index]))
            except ValueError:
                values.setdefault(col, []).append(np.nan)

    fp.close()

    if len(dates) == 0:
        raise ValueError('''
*
*  No data was collected from input.  Input must be a single line header with
*  comma separated column names, including a name for the date/time column.
*  The header must be followed by ISO 8601 formatted date string and comma
*  seperated columns of values.  Number of header columns must match number of
*  date/time stamp and value columns. For example:

Datetime, Flow_8603
2000-01-01 01:00:00 ,   4.5
2000-01-01 18:00:00 ,  10.8
2000-01-01 19:00:00 ,  11.1
...
''')

    for col in header[1:]:
        tmpres = pd.DataFrame(pd.Series(values[col],
                                        index=dates), columns=[col])
        try:
            result = result.join(tmpres)
        except NameError:
            result = tmpres

    if dense:
        result = result.asfreq(guess_freq(result)[1])
        return result

    return result


def read_excel_csv(fp, header=None):
    if header is not None:
        header = int(header)
    tsdata = pd.read_table(fp, header=header, sep=',', parse_dates=[0],
                           index_col=[0])
    return tsdata
