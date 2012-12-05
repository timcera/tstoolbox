
from cStringIO import StringIO

import pandas as pd
from dateutil.parser import parse

import numpy as np


def _isfinite(testval):
    '''
    Private utility for 'printiso' function.
    Just returns a blank in place of 'nan' so that other applications see just
    a missing value.
    '''
    if np.isfinite(testval):
        return str(testval)
    else:
        return ' '


# Utility
def printiso(tsd):
    '''
    Default output format for tstoolbox, wdmtoolbox, swmmtoolbox, and hspfbintoolbox.
    '''
    try:
        # Header
        print 'Datetime,', ', '.join( str(i) for i in tsd.columns )

        # Data
        for i in range(len(tsd)):
            print tsd.index[i], ', ', ', '.join( _isfinite(j) for j in tsd.values[i] )
    except IOError:
        return


def read_iso_ts(fp):
    '''
    Reads the format printed by 'print_iso'.
    '''
    header = fp.readline().split(',')
    header = [i.strip() for i in header]
    dates = []
    values = {}
    for line in fp.readlines():
        words = line.split(',')
        dates.append(parse(words[0]))
        for index,col in enumerate(header[1:]):
            try:
                values.setdefault(col, []).append(float(words[1 + index]))
            except ValueError:
                values.setdefault(col, []).append(float(np.nan))

    fp.close()
    print values.keys()
    for col in header[1:]:
        tmpres = pd.DataFrame(pd.Series(values[col],index=dates),
                 columns=[col])
        try:
            result = result.join(tmpres)
        except NameError:
            result = tmpres

    return result


def read_excel_csv(fp, header=None):
    if header <> None:
        header = int(header)
    tsdata = pd.read_table(fp, header=header, sep=',', parse_dates=[0],
            index_col=[0])
    return tsdata
