"""Collection of functions for the manipulation of time series."""

import os
import warnings

from pydantic import validate_arguments
from toolbox_utils import tsutils

warnings.filterwarnings("ignore")


@validate_arguments
@tsutils.doc(tsutils.docstrings)
def read(
    *filenames,
    force_freq=None,
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    skiprows=None,
    index_type="datetime",
    names=None,
    clean=False,
    source_units=None,
    target_units=None,
    round_index=None,
):
    """Combine time-series from different sources into single dataset.

    Prints the read in time-series in the tstoolbox standard format.

    WARNING: Accepts naive and timezone aware time-series by converting all to
    UTC and removing timezone information.

    Parameters
    ----------
    *filenames : str
        From the command line a list of space delimited filenames to read time
        series from.  Using the Python API a list or tuple of filenames.

        The supported file formats are CSV, Excel, WDM (Watershed Data
        Management), and HDF5.  The file formats are determined by the file
        extension.

        Comma-separated values (CSV) files or tab-separated values (TSV)::

            Separators will be automatically detected.  Columns can be
            selected by name or index, where the index for data columns starts
            at 1.

            CSV files requires a single line header of column names.  The
            default header is the first line of the input, but this can be
            changed for CSV files using the 'skiprows' option.

            Most common date formats can be used, but the closer to ISO 8601
            date/time standard the better.  ISO 8601 is roughly
            "YYYY-MM-DDTHH:MM:SS".

        Excel files (xls, xlsx, xlsm, xlsb, odf, ods, odt)::

            The time-series data is read in from one or more sheets.  The first
            row is assumed to be the header.  The first column is assumed to be
            the index.  The top left cell of the table should be the name of
            the date/time index and must be in cell A1.

        WDM files::

            One of more Data Set Numbers (DSN) can be specified in any order.

        HDF5 files (h5, hdf5, hdf)::

            One or more tables can be read from the HDF5 file.

        Command line examples:

            +--------------------------+------------------------------+
            | Keyword Example          | Description                  |
            +==========================+==============================+
            | fname.csv                | read all columns             |
            |                          | from 'fname.csv'             |
            +--------------------------+------------------------------+
            | fname.csv,2,1            | read data columns 2 and 1    |
            |                          | from 'fname.csv'             |
            +--------------------------+------------------------------+
            | fname.csv,2,skiprows=2   | read data column 2           |
            |                          | from 'fname.csv', skipping   |
            |                          | first 2 rows so header is    |
            |                          | read from third row          |
            +--------------------------+------------------------------+
            | fname.xlsx,2,Sheet21     | read all data from 2nd sheet |
            |                          | then all data from "Sheet21" |
            |                          | of 'fname.xlsx'              |
            +--------------------------+------------------------------+
            | fname.hdf5,Table12,T2    | read all data from table     |
            |                          | "Table12" then all data from |
            |                          | table "T2" of 'fname.hdf5'   |
            +--------------------------+------------------------------+
            | fname.wdm,210,110        | read DSNs 210, then 110      |
            |                          | from 'fname.wdm'             |
            +--------------------------+------------------------------+
            | -                        | read all columns from        |
            |                          | standard input (stdin)       |
            +--------------------------+------------------------------+

        Python library examples::

            Each entry in the list can be one of a pandas DataFrame, pandas
            Series, dict, tuple, list, StringIO, or file name with the options
            listed above.

            newdf = tstoolbox.read(['fname.csv,4,1', 'fname.xlsx', 'fname.hdf5'])

    ${force_freq}
        ${pandas_offset_codes}

    ${columns}

    ${start_date}

    ${end_date}

    ${dropna}

    ${skiprows}

    ${index_type}

    ${names}

    ${clean}

    ${source_units}

    ${target_units}

    ${float_format}

    ${round_index}

    ${tablefmt}
    """
    if force_freq is not None:
        dropna = "no"

    if isinstance(filenames, (list, tuple)) and len(filenames) == 1:
        filenames = filenames[0]

    isspacedelimited = any(
        not os.path.exists(str(fname))
        for fname in tsutils.make_list(filenames, sep=",")
    )

    if isspacedelimited:
        filenames = tsutils.make_list(filenames, sep=" ", flat=False)
    else:
        # All filenames are real files.  Therefore old style and just make
        # a simple list.
        filenames = tsutils.make_list(filenames, sep=",")
        warnings.warn(
            tsutils.error_wrapper(
                """
                Using "," separated files is deprecated in favor of space
                delimited files.
                """
            )
        )

    return tsutils.common_kwds(
        input_tsd=filenames,
        skiprows=skiprows,
        index_type=index_type,
        start_date=start_date,
        end_date=end_date,
        round_index=round_index,
        names=names,
        dropna=dropna,
        force_freq=force_freq,
        clean=clean,
        source_units=source_units,
        target_units=target_units,
        usecols=columns,
    )
