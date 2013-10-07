TSToolbox - Quick Guide
=======================
The tstoolbox is a Python script to manipulate time-series on the command
line.  Uses pandas (http://pandas.pydata.org/) or numpy (http://numpy.scipy.org) for any heavy lifting.

Requirements
------------
* pandas - on Windows this is part scientific Python distributions like
  Python(x,y), Anaconda, or Enthought.

* baker - command line parser

Installation
------------
Should be as easy as running ``pip install tstoolbox`` or ``easy_install
tstoolbox`` at any command line.  Not sure on Windows whether this will bring
in pandas, but as mentioned above, if you start with scientific Python distribution then you won't have a problem.

The tstoolbox script is actually made up of two modules, 'tstoolbox' which
handles all command line interaction and 'tsutil' which is a library of
functions that 'tstoolbox' uses.  This means that you can write your own
scripts to access ts files by importing the functionality from the 'tsutil' module.

Usage - Command Line
--------------------
Just run 'tstoolbox' to get a list of subcommands

.. program-output:: tstoolbox

The default for all of the subcommands is to accept data from stdin (typically
a pipe).  If a subcommand accepts an input file for an argument, you can use
"--infile=filename", or to expliticly specifiy from stdin "--infile='-'" .  

For the subcommands that output data it is printed to the screen and you can
then redirect to a file.

Sub-command Detail
------------------

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

equation
~~~~~~~~
.. program-output:: tstoolbox equation --help

fill
~~~~
.. program-output:: tstoolbox fill --help

filters
~~~~~~~
.. program-output:: tstoolbox filters --help

moving_window
~~~~~~~~~~~~~
.. program-output:: tstoolbox moving_window --help

peak_detection
~~~~~~~~~~~~~~
.. program-output:: tstoolbox peak_detection --help

pick
~~~~
.. program-output:: tstoolbox pick --help

plot
~~~~
.. program-output:: tstoolbox plot --help

print_test_data
~~~~~~~~~~~~~~~
.. program-output:: tstoolbox print_test_data --help

read
~~~~
.. program-output:: tstoolbox read --help

stdtozrxp
~~~~~~~~~
.. program-output:: tstoolbox stdtozrxp --help

tstopickle
~~~~~~~~~~
.. program-output:: tstoolbox tstopickle --help

Author
------
Tim Cera, P.E.

tim at cerazone dot net

Please send me a note if you find this useful, found a bug, submit a patch,
...etc.

