"""Collection of functions for the manipulation of time series."""

import cltoolbox
from cltoolbox.rst_text_formatter import RSTHelpFormatter
from pydantic import PositiveInt, validate_arguments
from sklearn.decomposition import PCA as skPCA
from toolbox_utils import tsutils


@cltoolbox.command("pca", formatter_class=RSTHelpFormatter)
@tsutils.doc(tsutils.docstrings)
def pca_cli(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    clean=False,
    skiprows=None,
    index_type="datetime",
    names=None,
    n_components=None,
    source_units=None,
    target_units=None,
    round_index=None,
    tablefmt="csv",
):
    """Return the principal components analysis of the time series.

    Does not return a time-series.

    Parameters
    ----------
    n_components : int
        [optional, default is None]

        The columns in the input_ts will be grouped into `n_components`
        groups.
    ${input_ts}
    ${columns}
    ${start_date}
    ${end_date}
    ${clean}
    ${skiprows}
    ${index_type}
    ${names}
    ${source_units}
    ${target_units}
    ${round_index}
    ${tablefmt}
    """
    tsutils.printiso(
        pca(
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            clean=clean,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            n_components=n_components,
            source_units=source_units,
            target_units=target_units,
            round_index=round_index,
        ),
        tablefmt=tablefmt,
    )


@validate_arguments
@tsutils.copy_doc(pca_cli)
def pca(
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    clean=False,
    skiprows=None,
    index_type="datetime",
    names=None,
    n_components: PositiveInt = None,
    source_units=None,
    target_units=None,
    round_index=None,
):
    """Return the principal components analysis of the time series."""

    tsd = tsutils.common_kwds(
        input_ts,
        skiprows=skiprows,
        names=names,
        index_type=index_type,
        start_date=start_date,
        end_date=end_date,
        round_index=round_index,
        pick=columns,
        source_units=source_units,
        target_units=target_units,
        clean=clean,
    )

    npca = skPCA(n_components)
    npca.fit(tsd.dropna(how="any"))
    return npca.components_
