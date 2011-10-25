#!/sjr/beodata/local/python_linux/bin/python
'''
'''
import sys

import scikits.timeseries as ts
import numpy as np
import baker

import tsutils

# Errors
class DSNDoesNotExist(Exception):
    pass

class FrequencyDoesNotMatchError(Exception):
    pass

class MissingValuesInInputError(Exception):
    pass

def _find_gcf(dividend, divisor):
    remainder = -1
    while remainder != 0:
        qoutient = dividend/divisor
        remainder = dividend%divisor
        if remainder != 0:
            dividend = divisor
            divisor = remainder
    gcf = divisor
    return divisor

@baker.command
def convertstdtoswmm(infile='-'):
    ''' Prints out data to the screen in a format that SWMM understands
    :param infile: Filename with data in 'ISOdate,value' format or '-' for stdin.
    '''
    tsd = tsutils.read_iso_ts(baker.openinput(infile))

    isodatestr = '01/01/{0.year:4d} 00:00:00'
    if tsd.freq >= 3000:
        isodatestr = '{0.month:02d}/01/{0.year:4d} 00:00:00'
    if tsd.freq >= 6000:
        isodatestr = '{0.month:02d}/{0.day:02d}/{0.year:4d} 00:00:00'
    if tsd.freq >= 7000:
        isodatestr = '{0.month:02d}/{0.day:02d}/{0.year:4d} {0.hour:02d}:00:00'
    if tsd.freq >= 8000:
        isodatestr = '{0.month:02d}/{0.day:02d}/{0.year:4d} {0.hour:02d}:{0.minute:02d}:00'
    if tsd.freq >= 9000:
        isodatestr = '{0.month:02d}/{0.day:02d}/{0.year:4d} {0.hour:02d}:{0.minute:02d}:{0.second:02d}'
    isodatestr = isodatestr + ' {1}'
    for i in range(len(tsd)):
        print isodatestr.format(tsd.dates[i], tsd[i])

@baker.command
def convertexcelcsvtostd(infile='-'):
    ''' Prints out data to the screen in a WISKI ZRXP format.
    :param infile: Filename with data in 'ISOdate,value' format or '-' for stdin.
    '''
    tsd = tsutils.read_excel_csv(baker.openinput(infile))
    tsutils.printiso(tsd)

@baker.command
def converttozrxp(infile='-'):
    ''' Prints out data to the screen in a WISKI ZRXP format.
    :param infile: Filename with data in 'ISOdate,value' format or '-' for stdin.
    '''
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    for i in range(len(tsd)):
        print '{0.year:04d}{0.month:02d}{0.day:02d}{0.hour:02d}{0.minute:02d}{0.second:02d} {1}'.format(tsd.dates[i], tsd[i])

#@baker.command
#def converttocsv(infile='-'):
#    ''' Prints out data to the screen in a CSV format.
#    :param infile: Filename with data in 'ISOdate,value' format or '-' for stdin.
#    '''
#    tsd = tsutils.read_iso_ts(baker.openinput(infile))
#    for i in range(len(tsd)):
#        print '{0.year:04d}{0.month:02d}{0.day:02d}{0.hour:02d}{0.minute:02d}{0.second:02d},{1}'.format(tsd.dates[i], tsd[i])

def tstopickle(wdmpath, dsn, filename, start_date=None, end_date=None):
    ''' Pickles the DSN data into a Python pickled file.  Can be brought back
    into Python with 'pickle.load' or 'numpy.load'.
    :param wdmpath: Path and WDM filename (<64 characters).
    :param dsn:     The Data Set Number in the WDM file.
    :param filename: The filename to store the pickled data.
    :param start_date: If not given defaults to start of data set.
    :param end_date:   If not given defaults to end of data set.
    '''
    tsd = wdm.read_dsn(wdmpath, int(dsn), start_date=start_date, end_date=end_date)
    tsd.dump(filename)

@baker.command
def converttoexcelcsv(infile='-', start_date=None, end_date=None):
    ''' Prints out data to the screen in a CSV format compatible with Excel.
    :param infile: Input comma separated file with 'ISO-8601_date/time,value'.
    :param start_date: If not given defaults to start of data set.
    :param end_date:   If not given defaults to end of data set.
    '''
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    if not start_date:
        start_date = tsd.dates[0]
    if not end_date:
        end_date = tsd.dates[-1]
    b = ts.date_array(start_date=start_date, end_date=end_date, freq=tsd.freq)
    tsd = tsd[b]
    xcldatestr = '{0.year:4d}/01/01 00:00:00'
    if tsd.freq >= 3000:
        xcldatestr = '{0.year:04d}/01/{0.day:2d} 00:00:00'
    if tsd.freq >= 6000:
        xcldatestr = '{0.year:04d}/{0.month:02d}/{0.day:2d} 00:00:00'
    if tsd.freq >= 7000:
        xcldatestr = '{0.year:04d}/{0.month:02d}/{0.day:2d} {0.hour:02d}:00:00'
    if tsd.freq >= 8000:
        xcldatestr = '{0.year:04d}/{0.month:02d}/{0.day:2d} {0.hour:02d}:{0.minute:02d}:00'
    if tsd.freq >= 9000:
        xcldatestr = '{0.year:04d}/{0.month:02d}/{0.day:2d} {0.hour:02d}:{0.minute:02d}:{0.second:02d}'
    xcldatestr = xcldatestr + ',{1}'
    for i in range(len(tsd)):
        print xcldatestr.format(tsd.dates[i], tsd[i][0])

@baker.command
def centered_moving_window(infile='-', span=2, start_date=None, end_date=None, statistic='mean'):
    ''' Calculates a centered moving window statistic.
    :param infile: Input comma separated file with 'ISO-8601_date/time,value'.
    :param span: The number of previous intervals to include in the calculation of the statistic.
    :param start_date: If not given defaults to start of data set.
    :param end_date:   If not given defaults to end of data set.
    :param statistic: 'mean' is the only option to calculate the centered moving window aggregation
    '''
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    if not start_date:
        start_date = tsd.dates[0]
    if not end_date:
        end_date = tsd.dates[-1]
    b = ts.date_array(start_date=start_date, end_date=end_date, freq=tsd.freq)
    tsd = tsd[b]
    import scikits.timeseries.lib as tslib
    span = int(span)
    if statistic == 'mean':
        newts = tslib.cmov_mean(tsd, span)
    else:
        print 'statistic ', statistic, ' is not implemented.'
        sys.exit()
    tsutils.printiso(newts)

@baker.command
def moving_window(infile='-', span=2, start_date=None, end_date=None, statistic='mean'):
    ''' Calculates a moving window statistic.
    :param infile: Input comma separated file with 'ISO-8601_date/time,value'.
    :param span: The number of previous intervals to include in the calculation of the statistic.
    :param start_date: If not given defaults to start of data set.
    :param end_date:   If not given defaults to end of data set.
    :param statistic: 'mean', 'mean_expw', 'sum', 'minimum', 'maximum', 'median', 'stdev', 'variance' to calculate the aggregation, defaults to 'mean'.
    '''
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    if not start_date:
        start_date = tsd.dates[0]
    if not end_date:
        end_date = tsd.dates[-1]
    b = ts.date_array(start_date=start_date, end_date=end_date, freq=tsd.freq)
    tsd = tsd[b]
    import scikits.timeseries.lib as tslib
    span = int(span)
    if statistic == 'mean':
        newts = tslib.mov_mean(tsd, span)
    elif statistic == 'mean_expw':
        newts = tslib.mov_average_expw(tsd, span)
    elif statistic == 'sum':
        newts = tslib.mov_sum(tsd, span)
    elif statistic == 'minimum':
        newts = tslib.mov_min(tsd, span)
    elif statistic == 'maximum':
        newts = tslib.mov_max(tsd, span)
    elif statistic == 'median':
        newts = tslib.mov_median(tsd, span)
    elif statistic == 'stdev':
        newts = tslib.mov_std(tsd, span)
    elif statistic == 'variance':
        newts = tslib.mov_var(tsd, span)
    else:
        print 'statistic ', statistic, ' is not implemented.'
        sys.exit()
    tsutils.printiso(newts)

@baker.command
def aggregate(infile='-', start_date=None, end_date=None, statistic='mean', agg_interval='daily'):
    ''' Takes a time series and aggregates to specified frequency, outputs to 'ISO-8601date,value' format.
    :param infile: Input comma separated file with 'ISO-8601_date/time,value'.
    :param start_date: If not given defaults to start of data set.
    :param end_date:   If not given defaults to end of data set.
    :param statistic: 'mean', 'sum', 'minimum', 'maximum', 'median', 'instantaneous' to calculate the aggregation, defaults to 'mean'.
    :param agg_interval: The 'hourly', 'daily', 'monthly', or 'yearly' aggregation intervals, defaults to daily.
    '''
    statd = {
            'mean': np.ma.mean,
            'sum': np.ma.sum,
            'min': np.ma.minimum,
            'max': np.ma.maximum,
            'median': np.ma.median,
            'instantaneous': ts.first_unmasked_val,
            }
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    if not start_date:
        start_date = tsd.dates[0]
    if not end_date:
        end_date = tsd.dates[-1]
    b = ts.date_array(start_date=start_date, end_date=end_date, freq=tsd.freq)
    tsd = tsd[b]

    newts = ts.convert(tsd, agg_interval, func=statd[statistic])
    tsutils.printiso(newts)

@baker.command
def calculate_fdc(infile='-', xdata_type = 'norm'):
    ''' Returns the frequency distrbution curve.  DOES NOT return a time-seris.
    :param infile: Filename with data in 'ISOdate,value' format or '-' for stdin.
    '''
    from scipy.stats.distributions import norm

    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    n = tsd.count(axis=-1)
    a = 1./(n + 1)
    b = 1 - a
    plotpos = np.ma.empty(len(tsd), dtype=float)
    if xdata_type == 'norm':
        plotpos[:n] = norm.ppf(np.linspace(a, b, n))
    if xdata_type == 'lin':
        plotpos[:n] = np.linespace(a, b, n)
    ydata = np.ma.sort(tsd, endwith=False)[::-1]
    xlabel = norm.cdf(plotpos)
    for x,y,z in zip(plotpos, ydata, xlabel):
        print x, y, z

@baker.command
def plot(infile='-', ofilename='plot.png', start_date=None, end_date=None, xtitle='Time', ytitle='', title='', figsize=(10,6.5)):
    ''' Time series plot.
    :param infile: Filename with data in 'ISOdate,value' format or '-' for stdin.
    :param ofilename: Output filename for the plot.  Extension defines the type, ('.png').
    :param start_date: If not given defaults to start of data set.
    :param end_date:   If not given defaults to end of data set.
    :param xtitle: Title of x-axis, defaults to 'Time'.
    :param ytitle: Title of y-axis, defaults to 'Flow'.
    :param title: Title of chart, defaults to ''.
    :param figsize: The (width, height) of plot as inches.  Defaults to (10,6.5).
    '''
    import matplotlib
    matplotlib.use('Agg')
    import pylab as pl
    #import scikits.timeseries.lib.plotlib as pl
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    if not start_date:
        start_date = tsd.dates[0]
    if not end_date:
        end_date = tsd.dates[-1]
    b = ts.date_array(start_date=start_date, end_date=end_date, freq=tsd.freq)
    tsd = tsd[b]
    fig = pl.figure(figsize=figsize)
    fsp = fig.add_subplot(111)
    fsp.plot(tsd.dates, tsd.data)
    fsp.set_xlabel(xtitle)
    fsp.set_ylabel(ytitle)
    fsp.set_title(title)
    fsp.xaxis_date()
    fig.savefig(ofilename)

def plotcalibandobs(mwdmpath, mdsn, owdmpath, odsn, ofilename):
    ''' IN DEVELOPMENT Plot model and observed data.
    :param mwdmpath: Path and WDM filename with model data (<64 characters).
    :param mdsn: DSN that contains the model data.
    :param owdmpath: Path and WDM filename with obs data (<64 characters).
    :param mdsn: DSN that contains the observed data.
    :param ofilename: Output filename for the plot.
    '''
    import scikits.hydroclimpy.plotlib as cpl
    # Create hydrograph plot
    hydroargs = {'color': 'blue',
                 'lw': 0.5,
                 'ls': 'steps-post',
                }
    hyetoargs = {'c': 'blue',
                 'lw': 0.5,
                 'ls': 'steps-post',
                }
    fig = cpl.hydrograph(rain['carr'], obs['carr'], figsize=figsize, hydroargs=hydroargs, hyetoargs=hyetoargs)
    # The following just plots 1 dot - I need the line parameters in lin1 to make the legend
    lin1 = fig.hydro.plot(obs['carr'].dates[0:1], obs['carr'][0:1], color='blue', ls='-', lw=0.5, label=line[11])
    rlin1 = fig.hyeto.plot(rain['carr'].dates[0:1], rain['carr'][0:1], color='blue', ls='-', lw=0.5, label=line[9])
    lin2 = fig.hydro.plot(sim['carr'].dates, sim['carr'], color='red', ls=':', lw=1, label='Simulated')
    fig.hyeto.set_ylabel("Total Daily\nRainfall (inches)")
    fig.hyeto.set_ylim(ymax = max_daily_rain, ymin = 0)
    fig.hydro.set_ylabel("Average Daily {0} {1}".format(line[14], parunitmap[line[14]]))
    if min(obs['arr']) > 0 and line[14] == 'Flow':
        fig.hydro.set_ylim(ymin = 0)
    #fig.hydro.set_yscale("symlog", linthreshy=100)
    fig.hydro.set_dlim(plot_start_date, plot_end_date)
    fig.hydro.set_xlabel("Date")
    #fig.legend((lin1, lin2), (line[11], 'Simulated'), (0.6, 0.55))
    make_legend(fig.hydro, (lin1[-1], lin2[-1]))
    make_legend(fig.hyeto, (rlin1[-1],))
    fig.savefig(newbasename + "_daily_hydrograph.png")
    fig.savefig(rnewbasename + "_daily_hydrograph.png")

baker.run()

