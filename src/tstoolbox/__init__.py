#!/sjr/beodata/local/python_linux/bin/python
'''
tstoolbox is a collection of command line tools for the manipulation of time
series.
'''
import sys
import os.path

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


def _sniff_filetype(filename):

    # Is it a pickled file...
    try:
        return pd.core.common.load(filename)
    except:
        pass

    # Really hard to determine if there is a header -
    # Assume yes...
    return pd.read_table(open(filename, 'rb'), header=0, sep=',',
            parse_dates=[0], index_col=[0])


def _print_input(iftrue, input, output, suffix):
    if suffix:
        output = output.rename(columns=lambda xloc: xloc + suffix)
    if iftrue:
        tsutils.printiso(input.join(output))
    else:
        tsutils.printiso(output)


@baker.command
def read(*filenames):
    '''
    Collect time series from a list of pickle or csv files then print
    in the tstoolbox standard format.
    :param filenames: List of filenames to read time series from.
    '''
    for filename in filenames:
        fname = os.path.basename(os.path.splitext(filename)[0])
        tsd = _sniff_filetype(filename)
        if len(filenames) > 1:
            tsd = tsd.rename(columns=lambda x:fname + '_' + x)
        try:
            result = result.join(tsd)
        except NameError:
            result = tsd
    tsutils.printiso(result)


@baker.command
def date_slice(start_date=None, end_date=None, infile='-'):
    '''
    Prints out data to the screen between start_date and end_date
    :param start_date: The start_date of the series in ISOdatetime format, or
        'None' for beginning.
    :param end_date: The end_date of the series in ISOdatetime format, or
        'None' for end.
    :param infile: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    tsutils.printiso(_date_slice(infile=infile, start_date=start_date,
        end_date=end_date))



@baker.command
def peak_detection(window=24, type='peak', method='rel', print_input=False,
        infile='-'):
    '''
    Peak and valley detection.
    :param infile: Filename with data in 'ISOdate,value' format or '-' for
        stdin.  Default is stdin.
    :param type: 'peak', 'valley', or 'both' to determine what should be
        returned.  Default is 'peak'.
    :param method: 'rel', 'minmax', 'zero_crossing' methods are available.
        You can try different ones, but 'rel' is the default and is likely the
        best.
    :param window: There will not usually be multiple peaks within the window
        number of values.  The different `method`s use this variable in different
        ways.
    :param print_input: If set to 'True' will include the input columns in the
        output table.  Default is 'False'.
    '''
    if type not in ['peak', 'valley', 'both']:
        raise ValueError("The `type` argument must be one of 'peak', 'valley', or 'both'.  You supplied {0}.".format(type))
    if method not in ['rel', 'minmax', 'zero_crossing']:
        raise ValueError("The `method` argument must be one of 'minmax', 'zero_crossing'.  You supplied {0}.".format(type))

    tsd = tsutils.read_iso_ts(baker.openinput(infile))

    if method == 'rel':
        func = tsutils._argrel
        window = int(window/2)
        if window == 0:
            window = 1
    elif method == 'minmax':
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

    _print_input(print_input, tsd, tmptsd, None)


@baker.command
def convert(factor=1.0, offset=0.0, print_input=False, infile='-'):
    '''
    Converts values of a time series by applying a factor and offset.  See the
        'equation' subcommand for a generalized form of this command.
    :param factor: Factor to multiply the time series values.
    :param offset: Offset to add to the time series values.
    :param print_input: If set to 'True' will include the input columns in the
        output table.  Default is 'False'.
    :param infile: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    tmptsd = tsd*factor + offset
    _print_input(print_input, tsd, tmptsd, '_convert')


@baker.command
def equation(equation, print_input=False, infile='-'):
    '''
    Applies <equation> to the time series data.  The <equation> argument is a
        string contained in single quotes with 'x' used as the variable
        representing the input.  For example, '(1 - x)*np.sin(x)'
    :param equation: String contained in single quotes that defines the
        equation.  The input variable place holder is 'x'.  Mathematical
        functions in the 'np' (numpy) name space can be used.  For example,
        'x*4 + 2', 'x**2 + np.cos(x)', and 'np.tan(x*np.pi/180)' are all valid
        <equation> strings.
    :param print_input: If set to 'True' will include the input columns in the
        output table.  Default is 'False'.
    :param infile: Filename with data in 'ISOdate,value' format or '-' for
        stdin.
    '''
    x = tsutils.read_iso_ts(baker.openinput(infile))
    y = eval(equation)
    _print_input(print_input, x, y, '_equation')


@baker.command
def stdtozrxp(rexchange=None, infile='-'):
    '''
    Prints out data to the screen in a WISKI ZRXP format.
    :param rexchange: The REXCHANGE ID to be written inro the zrxp header.
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
    Python with 'pickle.load' or 'numpy.load'.  See also 'tstoolbox read'.
    :param filename: The filename to store the pickled data.
    '''
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    pd.core.common.save(tsd, filename)


@baker.command
def moving_window(span=2, statistic='mean', print_input=False, infile='-'):
    '''
    Calculates a moving window statistic.
    :param infile: Input comma separated file with 'ISO-8601_date/time,value'.
    :param span: The number of previous intervals to include in the
        calculation of the statistic.
    :param statistic: 'mean', 'kurtosis', 'median', 'skew', 'stdev', 'sum',
        'variance', 'expw_mean', 'expw_stdev', 'expw_variance', 'minimum',
        'maximum' to calculate the aggregation, defaults to 'mean'.
    :param print_input: If set to 'True' will include the input columns in the
        output table.  Default is 'False'.
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
    _print_input(print_input, tsd, newts, '_' + statistic)


@baker.command
def aggregate(statistic='mean', agg_interval='daily', print_input=False, infile='-'):
    '''
    Takes a time series and aggregates to specified frequency, outputs to
        'ISO-8601date,value' format.
    :param infile: Input comma separated file with 'ISO-8601_date/time,value'.
    :param statistic: 'mean', 'sum', 'minimum', 'maximum', 'median',
        'instantaneous' to calculate the aggregation, defaults to 'mean'.
    :param agg_interval: The 'hourly', 'daily', 'monthly', or 'yearly'
        aggregation intervals, defaults to daily.
    :param print_input: If set to 'True' will include the input columns in the
        output table.  Default is 'False'.
    '''
    aggd = {'hourly': 'H',
            'daily': 'D',
            'monthly': 'M',
            'yearly': 'A'
            }
    tsd = tsutils.read_iso_ts(baker.openinput(infile))
    newts = tsd.resample(aggd[agg_interval], how=statistic)
    _print_input(print_input, tsd, newts, '_' + statistic)


@baker.command
def calculate_fdc(x_plotting_position = 'norm', infile='-'):
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

