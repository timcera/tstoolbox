Changelog
=========

0.4.1
Date:   Fri Dec 21 07:58:16 2012 -0500

    * 0.4.1 release - better 'equation' when index is negative

0.4
Date:   Fri Dec 21 07:55:49 2012 -0500

    * Added 'equation' subcommand to manipulate time-series.
    * Added 'read' subcommand to read in multiple files at once.
    * Finished 'peak_detection' subcommand.
    * Added '--print_input' option to those subcommands where it made sense.

0.3.1
Date: 2012-12-05

    * Removed print statement used for debugging.

0.3
Date: 2012-12-05

    * Removed everything from scikits.timeseries to pandas.
    * Took as much advantage of pandas as I could to streamline many of the
      sub-commands.
    * Added 'date_slice' sub-command and started to remove the
      '--start_date' and '--end_date' options from other commands.  Use
      'date_slice' instead in the pipe.


0.2.1
Date: 2011-10-25

    * Fixed how the start_date and end_date keywords were handled.  Minor
      bug fixes and code cleanups.

0.2
Date: 2011-10-21

    * The scikits.timeseries was throwing a segmentation fault.  Unproven,
      but suspect it was because of the latest numpy.  Only developed a
      work-around - eventually will move to the new datetime64 in numpy 1.7
      and greater.

0.1
Date: 2011-04-22

    * Initial release

