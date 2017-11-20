.. image:: https://travis-ci.org/timcera/tstoolbox.svg?branch=master
    :target: https://travis-ci.org/timcera/tstoolbox
    :height: 20

.. image:: https://coveralls.io/repos/timcera/tstoolbox/badge.png?branch=master
    :target: https://coveralls.io/r/timcera/tstoolbox?branch=master
    :height: 20

.. image:: https://img.shields.io/pypi/v/tstoolbox.svg
    :alt: Latest release
    :target: https://pypi.python.org/pypi/tstoolbox

.. image:: http://img.shields.io/badge/license-BSD-lightgrey.svg
    :alt: tstoolbox license
    :target: https://pypi.python.org/pypi/tstoolbox/

TSToolbox - Quick Guide
=======================
The tstoolbox is a Python script to manipulate time-series on the command line
or by function calls within Python.  Uses pandas (http://pandas.pydata.org/)
or numpy (http://numpy.scipy.org) for any heavy lifting.

Requirements
------------
* pandas - on Windows this is part scientific Python distributions like
  Python(x,y), Anaconda, or Enthought.

* mando - command line parser

Installation
------------
Should be as easy as running ``pip install tstoolbox`` or ``easy_install
tstoolbox`` at any command line.  Not sure on Windows whether this will bring
in pandas, but as mentioned above, if you start with scientific Python
distribution then you shouldn't have a problem.

Usage - Command Line
--------------------
Just run 'tstoolbox --help' to get a list of subcommands

usage: tstoolbox [-h]
                 {fill,about,createts,filter,read,date_slice,describe,peak_detection,convert,equation,pick,stdtozrxp,tstopickle,accumulate,rolling_window,aggregate,replace,clip,add_trend,remove_trend,calculate_fdc,stack,unstack,plot,dtw,pca,normalization,converttz,convert_index_to_julian,pct_change,rank,date_offset}
                 ...

    about               
        Display version number and system information.

    accumulate          
        Calculate accumulating statistics.

    add_trend           
        Add a trend.

    aggregate           
        Take a time series and aggregate to specified frequency.

    calculate_fdc       
        Return the frequency distribution curve.

    clip                
        Return a time-series with values limited to [a_min, a_max].

    convert             
        Convert values of a time series by applying a factor and offset.

    convert_index_to_julian 
        Convert date/time index to Julian dates from different epochs.

    converttz           
        Convert the time zone of the index.

    createts            
        Create empty time series, optionally fill with a value.

    date_offset         
        Apply an offset to a time-series.

    date_slice          
        Print out data to the screen between start_date and end_date.

    describe            
        Print out statistics for the time-series.

    dtw                 
        Dynamic Time Warping.

    equation            
        Apply <equation_str> to the time series data.

    fill                
        Fill missing values (NaN) with different methods.

    filter              
        Apply different filters to the time-series.

    normalization       
        Return the normalization of the time series.

    pca                 
        Return the principal components analysis of the time series.

    pct_change          
        Return the percent change between times.

    peak_detection      
        Peak and valley detection.

    pick                
        Will pick a column or list of columns from input.

    plot                
        Plot data.

    rank                
        Compute numerical data ranks (1 through n) along axis.

    read                
        Collect time series from a list of pickle or csv files.

    remove_trend        
        Remove a 'trend'.

    replace             
        Return a time-series replacing values with others.

    rolling_window      
        Calculate a rolling window statistic.

    stack               
        Return the stack of the input table.

    stdtozrxp           
        Print out data to the screen in a WISKI ZRXP format.

    tstopickle          
        Pickle the data into a Python pickled file.

    unstack             
        Return the unstack of the input table.

The default for all of the subcommands is to accept data from stdin (typically
a pipe).  If a subcommand accepts an input file for an argument, you can use
"--input_ts=input_file_name.csv", or to explicitly specify from stdin (the
default) "--input_ts='-'".

For the subcommands that output data it is printed to the screen and you can
then redirect to a file.

Usage - API
-----------
You can use all of the command line subcommands as functions.  The function
signature is identical to the command line subcommands.  The return is always
a PANDAS DataFrame.  Input can be a CSV or TAB separated file, or a PANDAS
DataFrame and is supplied to the function via the 'input_ts' keyword.

Simply import tstoolbox::

    from tstoolbox import tstoolbox

    # Then you could call the functions
    ntsd = tstoolbox.fill(method='linear', input_ts='tests/test_fill_01.csv')

    # Once you have a PANDAS DataFrame you can use that as input to other 
    # tstoolbox functions.
    ntsd = tstoolbox.aggregate(statistic='mean', agg_interval='daily', input_ts=ntsd)

