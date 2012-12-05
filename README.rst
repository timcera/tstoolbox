TSTOOLBOX
=========
The tstoolbox is a Python script to manipulate time-series on the command
line.  Based on pandas (http://pandas.pydata.org/)

Requirements
============

Python requirements for all platforms
-------------------------------------
* pandas - on Windows this is part of the Python(x,y) distribution
  (http://code.google.com/p/pythonxy/)

* baker - command line parser

Installation
============

Should be as easy as running ``easy_install tstoolbox`` or ``pip install
tstoolbox`` at any command line.  Not sure on Windows whether this will bring
in pandas, but as mentioned above, if you start with Python(x,y)
then you won't have a problem.

The tstoolbox script is actually made up of two parts, 'tstoolbox.py' which
handles all command line interaction and 'tsutil.py' which is a library of
functions that 'tstoolbox.py' uses.  This means that you can write your own
scripts to access ts files by importing the functionality from 'tsutil.py'.

Running
=======
Just run 'tstoolbox.py' to get a list of subcommands::

    Usage: /sjr/beodata/local/bin/tstoolbox COMMAND <options>
    
    Available commands:
     aggregate      Takes a time series and aggregates to specified frequency,
                    outputs to 'ISO-8601date,value' format.
     calculate_fdc  Returns the frequency distrbution curve. DOES NOT return a
                    time-series.
     convert        Converts values of a time series by applying a factor and
                    offset.
     date_slice     Prints out data to the screen between start_date and
                    end_date
     excelcsvtostd  Prints out data to the screen in a tstoolbox 'standard' ->
                    ISOdate,value
     moving_window  Calculates a moving window statistic.
     pickletots     Reads in a time-series from a Python pickled file.
     plot           Time series plot. (10,6.5).
     stdtozrxp      Prints out data to the screen in a WISKI ZRXP format.
     tstopickle     Pickles the data into a Python pickled file. Can be brought
                    back into Python with 'pickle.load' or 'numpy.load'.
    
    Use '/sjr/beodata/local/bin/tstoolbox <command> --help' for individual command help.

If a subcommand accepts an input file for an argument, you can use '-' to
indicate that the input is from a pipe.  For the subcommands that output data
it is printed to the screen and you can then redirect to a file.

Author
======

Tim Cera, P.E.

tim at cerazone dot net

Please send me a note if you find this useful, found a bug, submit a patch,
...etc.

