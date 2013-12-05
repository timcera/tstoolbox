TSToolbox - Quick Guide
=======================
The tstoolbox is a Python script to manipulate time-series on the command line
or by function calls within Python.  Uses pandas (http://pandas.pydata.org/)
or numpy (http://numpy.scipy.org) for any heavy lifting.

Requirements
------------
* pandas - on Windows this is part scientific Python distributions like
  Python(x,y), Anaconda, or Enthought.

* baker - command line parser

Installation
------------
Should be as easy as running ``pip install tstoolbox`` or ``easy_install
tstoolbox`` at any command line.  Not sure on Windows whether this will bring
in pandas, but as mentioned above, if you start with scientific Python
distribution then you won't have a problem.

Usage - Command Line
--------------------
Just run 'tstoolbox' to get a list of subcommands

.. program-output:: tstoolbox

The default for all of the subcommands is to accept data from stdin (typically
a pipe).  If a subcommand accepts an input file for an argument, you can use
"--input_ts=input_file_name.csv", or to expliticly specify from stdin (the
default) "--input_ts='-'" .  

For the subcommands that output data it is printed to the screen and you can
then redirect to a file.

Sub-command Detail
''''''''''''''''''

accumulate
~~~~~~~~~~
.. program-output:: tstoolbox accumulate --help

aggregate
~~~~~~~~~
.. program-output:: tstoolbox aggregate --help

calculate_fdc
~~~~~~~~~~~~~
.. program-output:: tstoolbox calculate_fdc --help

convert
~~~~~~~
.. program-output:: tstoolbox convert --help

date_slice
~~~~~~~~~~
.. program-output:: tstoolbox date_slice --help

describe
~~~~~~~~
.. program-output:: tstoolbox describe --help

equation
~~~~~~~~
.. program-output:: tstoolbox equation --help

fill
~~~~
.. program-output:: tstoolbox fill --help

filter
~~~~~~
.. program-output:: tstoolbox filter --help

peak_detection
~~~~~~~~~~~~~~
.. program-output:: tstoolbox peak_detection --help

pick
~~~~
.. program-output:: tstoolbox pick --help

plot
~~~~
.. program-output:: tstoolbox plot --help

read
~~~~
.. program-output:: tstoolbox read --help

rolling_window
~~~~~~~~~~~~~~
.. program-output:: tstoolbox rolling_window --help

stdtozrxp
~~~~~~~~~
.. program-output:: tstoolbox stdtozrxp --help

tstopickle
~~~~~~~~~~
.. program-output:: tstoolbox tstopickle --help

Usage - API
-----------
You can use all of the command line subcommands as functions.  The function
signature is identical to the command line subcommands.  The return is always
a PANDAS DataFrame.  Input can be a CSV or TAB separated file, or a PANDAS
DataFrame and is supplied to the function via the 'input_ts' keyword.

Simply import tstoolbox::

    import tstoolbox

    # Then you could call the functions
    ntsd = tstoolbox.fill(method='linear', input_ts='tests/test_fill_01.csv')

    # Once you have a PANDAS DataFrame you can use that as input.
    ntsd = tstoolbox.aggregate(statistic='mean', agg_interval='daily', input_ts=ntsd)

Author
------
Tim Cera, P.E.

tim at cerazone dot net
