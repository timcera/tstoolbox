import sys
import re

import scikits.timeseries as ts

# Utility
def isoformatstr(date):
    ''' Returns the appropriate sized dates according to freq.'''
    isodatestr = '{0.year}'
    if date.freq >= 3000:
        isodatestr = isodatestr + '-{0.month:02d}'
    if date.freq >= 6000:
        isodatestr = isodatestr + '-{0.day:02d}'
    if date.freq >= 7000:
        isodatestr = isodatestr + 'T{0.hour:02d}'
    if date.freq >= 8000:
        isodatestr = isodatestr + ':{0.minute:02d}'
    if date.freq >= 9000:
        isodatestr = isodatestr + ':{0.second:02d}'
    return isodatestr

def printiso(ts):
    # Hopefully the date string will be the same for all date/times in the series
    dateformat = isoformatstr(ts.dates[0])
    for i in range(len(ts)):
        print '{0},{1}'.format(dateformat.format(ts.dates[i]), ts[i])

def dateconverter(datestr):
    words = re.findall(r'\d+', str(datestr))
    if len(words) == 1:
        tsdate = ts.Date(freq='yearly',
                         year=int(words[0]))
    if len(words) == 2:
        tsdate = ts.Date(freq='monthly',
                         year=int(words[0]),
                         month=int(words[1]))
    if len(words) == 3:
        tsdate = ts.Date(freq='daily',
                         year=int(words[0]),
                         month=int(words[1]),
                         day=int(words[2]))
    if len(words) == 4:
        tsdate = ts.Date(freq='hourly',
                         year=int(words[0]),
                         month=int(words[1]),
                         day=int(words[2]),
                         hour=int(words[3]))
    if len(words) == 5:
        tsdate = ts.Date(freq='minutely',
                         year=int(words[0]),
                         month=int(words[1]),
                         day=int(words[2]),
                         hour=int(words[3]),
                         minute=int(words[4]))
    if len(words) == 6:
        tsdate = ts.Date(freq='secondly',
                         year=int(words[0]),
                         month=int(words[1]),
                         day=int(words[2]),
                         hour=int(words[3]),
                         minute=int(words[4]),
                         second=int(words[5]))
    return tsdate

def read_iso_ts(fp):
    tsdata = ts.tsfromtxt(fp, delimiter=',', datecols=(0), dateconverter=dateconverter, usecols=(0,1), names='value')
    return tsdata['value']

