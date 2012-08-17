TSTOOLBOX
=========
The tstoolbox is a Python script to manipulate time-series.  Based on
scikits.timeseries.

Requirements
============

Python requirements for all platforms
-------------------------------------
* scikits.timeseries - on Windows this is part of the Python(x,y) distribution
  (http://code.google.com/p/pythonxy/)

* baker - command line parser

Installation
============

Should be as easy as running ``easy_install tstoolbox`` or ``pip install
tstoolbox`` at any command line.  Not sure on Windows whether this will bring
in scikits.timeseries, but as mentioned above, if you start with Python(x,y)
then you won't have a problem.

The tstoolbox script is actually made up of two parts, 'tstoolbox.py' which
handles all command line interaction and 'tsutil.py' which is a library of
functions that 'tstoolbox.py' uses.  This means that you can write your own
scripts to access ts files by importing the functionality from 'tsutil.py'.

Running
=======
Just run 'tstoolbox.py' to get a list of subcommands::

    Usage: /sjr/beodata/local/bin/tstoolbox.py COMMAND <options>
    
    Available commands:
    
     aggregate               Takes a time series and aggregates to specified
                             frequency, outputs to 'ISO-8601date,value' format.
     calculate_fdc           Returns the frequency distrbution curve. DOES NOT
                             return a time-seris.
     centered_moving_window  Calculates a centered moving window statistic.
     convertexcelcsvtostd    Prints out data to the screen in a WISKI ZRXP
                             format.
     convertstdtoswmm        Prints out data to the screen in a format that SWMM
                             understands
     converttoexcelcsv       Prints out data to the screen in a CSV format
                             compatible with Excel.
     converttozrxp           Prints out data to the screen in a WISKI ZRXP
                             format.
     moving_window           Calculates a moving window statistic.
     plot                    Time series plot.
    
    Use "/sjr/beodata/local/bin/tstoolbox.py <command> --help" for individual command help.

If a subcommand accepts an input file for an argument, you can use '-' to
indicate that the input is from a pipe.  For the subcommands that output data
it is printed to the screen and you can then redirect to a file.

Author
======

Tim Cera, P.E.

tim at cerazone dot net

Please send me a note if you find this useful, found a bug, submit a patch,
...etc.

