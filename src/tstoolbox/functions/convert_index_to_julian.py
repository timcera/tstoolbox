"""Collection of functions for the manipulation of time series."""

import warnings

from ..toolbox_utils.src.toolbox_utils import tsutils
from .convert_index import convert_index

warnings.filterwarnings("ignore")


@tsutils.doc(tsutils.docstrings)
def convert_index_to_julian(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    round_index=None,
    dropna="no",
    clean=False,
    index_type="datetime",
    names=None,
    source_units=None,
    target_units=None,
    skiprows=None,
):
    """DEPRECATED: Use convert_index instead.

    Will be removed in a future version of `tstoolbox`.

    Use `convert_index` in place of `convert_index_to_julian`.

    For command line::

        tstoolbox convert_index julian ...

    For Python::

        from tstoolbox import tstoolbox
        ndf = ntstoolbox.convert_index('julian', ...)
    """
    warnings.warn(
        """
*
*   DEPRECATED in favor of using `convert_index` with the 'julian'
*   option.
*
*   Will be removed in a future version of `tstoolbox`.
*
"""
    )
    return convert_index(
        "julian",
        columns=columns,
        input_ts=input_ts,
        start_date=start_date,
        end_date=end_date,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        skiprows=skiprows,
        names=names,
        source_units=source_units,
        target_units=target_units,
    )
