
from __future__ import print_function
from __future__ import division


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
    mapcode = {'A': 6,
               'AS': 6,
               'M': 5,
               'MS': 5,
               'D': 4,
               'H': 3,
               'T': 2,
               'S': 1
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
        return mapcode[
            pndcode[
                data.index.freq.nanos//1000000000]], data.index.freqstr, 1

    interval = np.unique(data.index.values[1:] -
                         data.index.values[:-1])//1000000000
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
            for seconds in sorted(pndcode, reverse=True):
                if seconds > inter:
                    continue
                if inter % seconds == 0:
                    accumulate_freq.append(seconds)
                    break

        accumulate_freq.sort()
        finterval = accumulate_freq[0]
    else:
        finterval = interval[0]

    # Calculate tstep
    tstep = 1
    try:
        pandacode = pndcode[finterval]
    except KeyError:
        # finterval is probably a multiple of an existing interval
        for seconds in sorted(pndcode, reverse=True):
            if finterval > seconds and finterval % seconds == 0:
                tstep = finterval//seconds
                pandacode = pndcode[seconds]
                break

    if pandacode == 'M':
        if data.index[0].day == 1:
            pandacode = 'MS'
    elif pandacode == 'A':
        if data.index[0].month == 1:
            pandacode = 'AS'

    return mapcode[pandacode], pandacode, tstep


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
    try:
        if tsd.index.is_all_dates:
            tsd.index.name = 'Datetime'
            tsd.to_csv(sys.stdout, float_format='%g')
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
            gf = guess_freq(indat)
            return indat.asfreq('{0}{1}'.format(gf[2], gf[1]))
        else:
            return indat

    fp = baker.openinput(indat)

    result = pd.io.parsers.read_table(fp, header=0, sep=',', index_col=0,
                                      parse_dates=True)

    result.index.name = 'Datetime'
    result.columns = [i.strip() for i in result.columns]

    if dense:
        gf = guess_freq(result)
        result = result.asfreq('{0}{1}'.format(gf[2], gf[1]))

    return result


def read_excel_csv(fp, header=None):
    if header is not None:
        header = int(header)
    tsdata = pd.read_table(fp, header=header, sep=',', parse_dates=[0],
                           index_col=[0])
    return tsdata
