"""Collection of functions for the manipulation of time series."""

from pydantic import PositiveInt, validate_arguments
from sklearn.decomposition import PCA as skPCA
from toolbox_utils import tsutils


@validate_arguments
@tsutils.doc(tsutils.docstrings)
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
