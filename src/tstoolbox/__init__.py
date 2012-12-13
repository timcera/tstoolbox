#!/sjr/beodata/local/python_linux/bin/python
'''
tstoolbox is a collection of command line tools for the manipulation of time
series.
'''
import sys

import pandas as pd
import numpy as np
import baker

import tsutils


def _date_slice(infile='-', start_date=None, end_date=None):
    '''
    Private function to slice time series.
    '''
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    return tsd[start_date:end_date]


@baker.command
def date_slice(infile='-', start_date=None, end_date=None):
    '''
    Prints out data to the screen between start_date and end_date
    :param infile: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    :param start_date: The start_date of the series in ISOdatetime format, or
        'None' for beginning.
    :param end_date: The end_date of the series in ISOdatetime format, or
        'None' for end.
    '''
    tsutils.printiso(_date_slice(infile=infile, start_date=start_date,
        end_date=end_date))


@baker.command
def excelcsvtostd(infile='-', header=0):
    '''
    Prints out data to the screen in a tstoolbox 'standard' -> ISOdate,value
    :param infile: Filename with data in 'ISOdate,value' format
    '''
    tsd = tsutils.read_excel_csv(baker.openinput(infile), header=header)
    tsutils.printiso(tsd)


@baker.command
def nppeak_detection(infile='-', print_input=False):
    '''
    Use scipy.signal peak detection...
    '''
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    argmax = tsutils._argrelmax(tsd)
    if print_input:
        tsutils.printiso(tsd.join(argmax))
    else:
        tsutils.printiso(argmax)

@baker.command
def peak_detection(infile='-', type='peak', method='minmax', window=24, print_input=False):
    '''
    Peak and valley detection.
    :param infile: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    :param type: 'peak', 'valley', or 'both' to determine what should be returned.
    :param window: There will not be multiple peaks within the window number of values.
    '''
    if type not in ['peak', 'valley', 'both']:
        raise ValueError("The `type` argument must be one of 'peak', 'valley', or 'both'.  You supplied {0}.".format(type))
    if method not in ['minmax', 'zero_crossing']:
        raise ValueError("The `method` argument must be one of 'minmax', 'zero_crossing'.  You supplied {0}.".format(type))

    tsd = tsutils.read_iso_ts(baker.openinput(infile))

    if method == 'minmax':
        func = tsutils._peakdetect
        window = int(window/2)
        if window == 0:
             window = 1
    elif method == 'zero_crossing':
        func = tsutils._peakdetect_zero_crossing
        if not window % 2:
            window = window + 1

    if type == 'peak':
        tmptsd = tsd.rename(columns=lambda x: x+'_peak', copy=True)
    if type == 'valley':
        tmptsd = tsd.rename(columns=lambda x: x+'_valley', copy=True)
    if type == 'both':
        tmptsd = tsd.rename(columns=lambda x: x+'_peak', copy=True)
        tmptsd = tmptsd.join(tsd.rename(columns=lambda x: x+'_valley', copy=True))

    for c in tmptsd.columns:
        maxpeak, minpeak = func(tmptsd[c].values, window=int(window))
        if c[-5:] == '_peak':
            datavals = maxpeak
        if c[-7:] == '_valley':
            datavals = minpeak

        maxx, maxy = zip(*datavals)
        tmptsd[c][:] = np.nan
        tmptsd[c][np.array(maxx)] = maxy

    if print_input:
        tsutils.printiso(tsd.join(tmptsd))
    else:
        tsutils.printiso(tmptsd)

# Following was replaced by much better algorithm...
#@baker.command
#def peak_detection(type=1, k=24, h=1.5, infile='-', print_input=False):
#    '''
#    Determines a significance of a peak.
#    :param type: 1 or 2, type of algorithm used.
#    :param infile: Filename with data in 'ISOdate,value' format or '-' for
#        stdin.
#    '''
#    tsd = tsutils.read_iso_ts(baker.openinput(infile))
#
#    tmptsd = tsd.rename(columns=lambda x: x+'_peak_detection', copy=True)
#    if type == 1:
#        for i in range(len(tsd.values))[k:-k]:
#            tmptsd.values[i] = (
#                    np.nanmax(tsd.values[i] - tsd.values[i-k:i], axis=0) +
#                    np.nanmax(tsd.values[i] - tsd.values[i+1:i+k+1], axis=0))/2.0
#        tmptsd.values[:k] = np.nan
#        tmptsd.values[-k:] = np.nan
#        tmptsd[tmptsd.values<=0] = np.nan
#        globalmean = tmptsd.mean()
#        globalstd = tmptsd.std()
#        tmptsd.values[(tmptsd.values - globalmean.values) <
#                (h*globalstd.values)] = np.nan
#        selvals = tsutils._argrelmax(tmptsd.fillna(0.0).values)
#        tmptsd.values[:,:] = np.nan
#        tmptsd.values[selvals] = tsd.values[selvals]
#        #tmptsd.values[pd.rolling_max(tmptsd.fillna(0), 2*k + 1) !=
#        #        tmptsd.values] = np.nan
#        #isfi = np.isfinite(tmptsd)
#        #tmptsd.values[isfi] = tsd.values[isfi]
#    if print_input:
#        tsutils.printiso(tsd.join(tmptsd))
#    else:
#        tsutils.printiso(tmptsd)


@baker.command
def convert(factor=1.0, offset=0.0, infile='-'):
    '''
    Converts values of a time series by applying a factor and offset.
    :param factor: Factor to multiply the time series values.
    :param offset: Offset to add to the time series values.
    :param infile: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    tsd = tsd*factor + offset
    tsutils.printiso(tsd)


@baker.command
def stdtozrxp(infile='-', rexchange=None):
    '''
    Prints out data to the screen in a WISKI ZRXP format.
    :param infile: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    if len(tsd.columns) > 1:
        raise ValueError('The "stdtozrxp" command can only accept a single '
                         'time-series, instead it is seeing {0}'.format(len(tsd.columns)))
    if rexchange:
        print '#REXCHANGE{0}|*|'.format(rexchange)
    for i in range(len(tsd)):
        print ('{0.year:04d}{0.month:02d}{0.day:02d}{0.hour:02d}'
               '{0.minute:02d}{0.second:02d}, {1}').format(tsd.index[i],
                       tsd[tsd.columns[0]][i])


@baker.command
def tstopickle(filename, infile='-'):
    '''
    Pickles the data into a Python pickled file.  Can be brought back into
    Python with 'pickle.load' or 'numpy.load'.
    :param filename: The filename to store the pickled data.
    '''
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    pd.core.common.save(tsd, filename)


@baker.command
def pickletots(filename):
    '''
    Reads in a time-series from a Python pickled file.
    :param filename: Path and filename to Python pickled file.
    '''
    tsd = pd.core.common.load(filename)
    tsutils.printiso(tsd)


@baker.command
def moving_window(infile='-', span=2, statistic='mean'):
    '''
    Calculates a moving window statistic.
    :param infile: Input comma separated file with 'ISO-8601_date/time,value'.
    :param span: The number of previous intervals to include in the
        calculation of the statistic.
    :param statistic: 'mean', 'kurtosis', 'median', 'skew', 'stdev', 'sum',
        'variance', 'expw_mean', 'expw_stdev', 'expw_variance', 'minimum',
        'maximum' to calculate the aggregation, defaults to 'mean'.
    '''
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    span = int(span)
    if statistic == 'mean':
        newts = pd.stats.moments.rolling_mean(tsd, span)
    elif statistic == 'kurtosis':
        newts = pd.stats.moments.rolling_kurt(tsd, span)
    elif statistic == 'median':
        newts = pd.stats.moments.rolling_median(tsd, span)
    elif statistic == 'skew':
        newts = pd.stats.moments.rolling_skew(tsd, span)
    elif statistic == 'stdev':
        newts = pd.stats.moments.rolling_std(tsd, span)
    elif statistic == 'sum':
        newts = pd.stats.moments.rolling_sum(tsd, span)
    elif statistic == 'variance':
        newts = pd.stats.moments.rolling_var(tsd, span)
    elif statistic == 'expw_mean':
        newts = pd.stats.moments.ewma(tsd, span=span)
    elif statistic == 'expw_stdev':
        newts = pd.stats.moments.ewmstd(tsd, span=span)
    elif statistic == 'expw_variance':
        newts = pd.stats.moments.ewmvar(tsd, span=span)
    elif statistic == 'minimum':
        newts = pd.stats.moments.rolling_min(tsd, span)
    elif statistic == 'maximum':
        newts = pd.stats.moments.rolling_max(tsd, span)
    else:
        print 'statistic ', statistic, ' is not implemented.'
        sys.exit()
    tsutils.printiso(newts)


@baker.command
def aggregate(infile='-', statistic='mean', agg_interval='daily'):
    '''
    Takes a time series and aggregates to specified frequency, outputs to
        'ISO-8601date,value' format.
    :param infile: Input comma separated file with 'ISO-8601_date/time,value'.
    :param statistic: 'mean', 'sum', 'minimum', 'maximum', 'median',
        'instantaneous' to calculate the aggregation, defaults to 'mean'.
    :param agg_interval: The 'hourly', 'daily', 'monthly', or 'yearly'
        aggregation intervals, defaults to daily.
    '''
    aggd = {'hourly': 'H',
            'daily': 'D',
            'monthly': 'M',
            'yearly': 'A'
            }
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    newts = tsd.resample(aggd[agg_interval], how=statistic)
    tsutils.printiso(newts)


@baker.command
def calculate_fdc(infile='-', x_plotting_position = 'norm'):
    '''
    Returns the frequency distrbution curve.  DOES NOT return a time-series.
    :param infile: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    :param x_plotting_position: 'norm' or 'lin'.  'norm' defines a x plotting position Defaults to 'norm'.
    '''
    from scipy.stats.distributions import norm

    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    if len(tsd.columns) > 1:
        raise ValueError("This function currently only works with one time-series at a time.  You gave it {0}".format(len(tsd.columns)))

    cnt = len(tsd.values)
    a_tmp = 1./(cnt + 1)
    b_tmp = 1 - a_tmp
    plotpos = np.ma.empty(len(tsd), dtype=float)
    if x_plotting_position == 'norm':
        plotpos[:cnt] = norm.ppf(np.linspace(a_tmp, b_tmp, cnt))
        xlabel = norm.cdf(plotpos)
    if x_plotting_position == 'lin':
        plotpos[:cnt] = np.linspace(a_tmp, b_tmp, cnt)
        xlabel = plotpos
    ydata = np.ma.sort(tsd[tsd.columns[0]].values, endwith=False)[::-1]
    for xdat, ydat, zdat in zip(plotpos, ydata, xlabel):
        print xdat, ydat, zdat


@baker.command
def plot(infile='-', ofilename='plot.png', xtitle='Time', ytitle='',
        title='', figsize=(10, 6.5)):
    '''
    Time series plot.
    :param infile: Filename with data in 'ISOdate,value' format or '-' for
       stdin.
    :param ofilename: Output filename for the plot.  Extension defines the
       type, ('.png'). Defaults to 'plot.png'.
    :param xtitle: Title of x-axis, defaults to 'Time'.
    :param ytitle: Title of y-axis, defaults to 'Flow'.
    :param title: Title of chart, defaults to ''.
    :param figsize: The (width, height) of plot as inches.  Defaults to
    (10,6.5).
    '''
    import matplotlib
    matplotlib.use('Agg')
    import pylab as pl
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    fig = pl.figure(figsize=figsize)
    fsp = fig.add_subplot(111)
    print tsd
    fsp.plot(tsd.index, tsd.values)
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
    # Create hydrograph plot
    hydroargs = {'color': 'blue',
                 'lw': 0.5,
                 'ls': 'steps-post',
                }
    hyetoargs = {'c': 'blue',
                 'lw': 0.5,
                 'ls': 'steps-post',
                }
    fig = cpl.hydrograph(rain['carr'], obs['carr'], figsize=figsize,
            hydroargs=hydroargs, hyetoargs=hyetoargs)
    # The following just plots 1 dot - I need the line parameters in lin1 to
    # make the legend
    lin1 = fig.hydro.plot(obs['carr'].dates[0:1], obs['carr'][0:1],
            color='blue', ls='-', lw=0.5, label=line[11])
    rlin1 = fig.hyeto.plot(rain['carr'].dates[0:1], rain['carr'][0:1],
            color='blue', ls='-', lw=0.5, label=line[9])
    lin2 = fig.hydro.plot(sim['carr'].dates, sim['carr'], color='red', ls=':',
            lw=1, label='Simulated')
    fig.hyeto.set_ylabel("Total Daily\nRainfall (inches)")
    fig.hyeto.set_ylim(ymax = max_daily_rain, ymin = 0)
    fig.hydro.set_ylabel("Average Daily {0} {1}".format(line[14],
        parunitmap[line[14]]))
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


def main():
    baker.run()

