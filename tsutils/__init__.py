
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
    if np.isfinite(testval):
        return str(testval)
    else:
        return ' '


# Utility
def print_input(iftrue, input, output, suffix):
    if suffix:
        output = output.rename(columns=lambda xloc: xloc + suffix)
    if iftrue:
        return printiso(input.join(output, how='outer'))
    else:
        return printiso(output)


def _printiso(tsd):
    ''' Separate so can use in tests.
    '''
    try:
        # Header
        print('Datetime,', ', '.join(str(i) for i in tsd.columns))

        # Data
        for i in range(len(tsd)):
            print(tsd.index[i], ', ', ', '.join(
                _isfinite(j) for j in tsd.values[i]))
    except IOError:
        return


def printiso(tsd, sparse=False):
    '''
    Default output format for tstoolbox, wdmtoolbox, swmmtoolbox,
    and hspfbintoolbox.
    '''
    import traceback
    import os.path
    baker_cli = False
    for i in traceback.extract_stack():
        if os.path.basename(i[0]) == 'baker.py':
            baker_cli = True
            break
    if baker_cli:
        _printiso(tsd)
    else:
        return tsd


def read_iso_ts(indat):
    '''
    Reads the format printed by 'print_iso'.
    '''
    import baker

    if isinstance(indat, pd.DataFrame):
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
        dates.append(parse(words[0]))
        for index, col in enumerate(header[1:]):
            try:
                values.setdefault(col, []).append(float(words[1 + index]))
            except ValueError:
                values.setdefault(col, []).append(np.nan)

    fp.close()
    for col in header[1:]:
        tmpres = pd.DataFrame(pd.Series(values[col],
                                        index=dates), columns=[col])
        try:
            result = result.join(tmpres)
        except NameError:
            result = tmpres

    return result


def read_excel_csv(fp, header=None):
    if header is not None:
        header = int(header)
    tsdata = pd.read_table(fp, header=header, sep=',', parse_dates=[0],
                           index_col=[0])
    return tsdata
