import numpy as np


def check_arrays(predicted, reference, pname="predicted", rname="reference"):
    """Generic check of input arrays."""
    pdims = predicted.shape
    rdims = reference.shape
    if not np.array_equal(pdims, rdims):
        raise ValueError(
            """
*
*   The {3} and {4} field dimensions do not match.
*       shape({3}) = {0}
*       shape({4}) = {1}
*       {3} type: {2}
*
""".format(
                pdims, rdims, type(predicted), pname, rname
            )
        )
