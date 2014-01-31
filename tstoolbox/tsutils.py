
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
        sdate = None
    else:
        sdate = pd.Timestamp(start_date)
    if end_date is None:
        edate = None
    else:
        edate = pd.Timestamp(end_date)
    return input_tsd[sdate:edate]


def guess_freq(data):
    # Another way to do this is to abuse PANDAS .asfreq.  Basically, how low
    # can you go and maintain the same number of values.
    mapcode = {'A':  6,  # annual
               'AS': 6,  # annual start
               'M':  5,  # month
               'MS': 5,  # month start
               'D':  4,  # day
               'H':  3,  # hour
               'T':  2,  # minute
               'S':  1   # second
               }

    pndcode = {31536000: 'A',  # 365 days
               31622400: 'A',  # 366 days
               2678400:  'M',  # 31 day month
               2592000:  'M',  # 30 day month
               2505600:  'M',  # 29 day month
               2419200:  'M',  # 28 day month
               604800:   'W',  # 7 day week
               86400:    'D',  # 1 day
               3600:     'H',  # 1 hour
               60:       'T',  # 1 minute
               1:        'S'   # 1 second
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


def _apply_across_columns(func, xtsd, **kwds):
    for col in xtsd.columns:
        xtsd[col] = func(xtsd[col], **kwds)
    return xtsd


def _printiso(tsd, date_format='%Y-%m-%d %H:%M:%S', delimiter=','):
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
    import csv

    if isinstance(indat, pd.DataFrame):
        result = indat
        if not dense:
            return result

    elif isinstance(indat, str):
        if indat == '-':
            # format must be the tstoolbox standard
            has_header = True
            dialect = csv.excel
            fp = baker.openinput(indat)
        elif os.path.exists(indat):
            # Is it a pickled file?
            try:
                result = pd.io.pickle.read_pickle(indat)
                fp = False
            except:
                # Maybe a CSV file?
                with open(indat) as csvfile:
                    readsome = csvfile.read(2048)
                    dialect = csv.Sniffer().sniff(readsome,
                                                  delimiters=', \t:|')
                    has_header = csv.Sniffer().has_header(readsome)

                fp = open(indat)

        if fp:
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
        gf = guess_freq(result)
        result = result.asfreq('{0}{1}'.format(gf[2], gf[1]))

    return result


def read_excel_csv(fp, header=None):
    if header is not None:
        header = int(header)
    tsdata = pd.read_table(fp, header=header, sep=',', parse_dates=[0],
                           index_col=[0])
    return tsdata
