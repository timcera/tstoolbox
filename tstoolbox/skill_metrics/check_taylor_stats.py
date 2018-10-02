import numpy as np


def check_taylor_stats(stds, crmsds, cors, threshold=0.01):
    """
    Check that input statistics satisfy Taylor diagram relation to <1%.

    Function terminates with an error if not satisfied. The threshold is
    the ratio of the difference between the statistical metrics and the
    centered root mean square difference:

     abs(crmsds^2 - (stds^2 + stds(1)^2 - 2*stds*stds(1)*cors))/crmsds^2

    Note that the first element of the statistics vectors must contain
    the value for the reference field.

    INPUTS:
    stds      : Standard deviations
    crmsds    : Centered Root Mean Square Difference
    cors      : Correlation
    threshold : limit for acceptance, e.g. 0.1 for 10% (default 0.01)

    OUTPUTS:
    None.

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com

    Created on Dec 3, 2016
    """
    if threshold < 1e-7:
        ValueError('threshold value must be positive: ' + str(threshold))

    diff = (np.square(crmsds[1:]) -
            (np.square(stds[1:]) +
             np.square(stds[0]) -
             2.0 * stds[0] * np.multiply(stds[1:], cors[1:])))
    diff = np.abs(np.divide(diff, np.square(crmsds[1:])))
    index = np.where(diff > threshold)
    if len(index) > 0:
        ii = np.where(diff != 0)
        if len(ii) == len(diff):
            ValueError("""
*
*   Incompatible data
*
*   You must have:
*       crmsds - sqrt(stds.^2 + stds(1)^2 - 2*stds*stds(1).*cors) = 0
*
""")
        else:
            ValueError("""
*
*   Incompatible data indices: {0}
*
*   You must have:
*       crmsds - sqrt(stds.^2 + stds(1)^2 - 2*stds*stds(1).*cors) = 0
*
""".format(ii))

    return diff
