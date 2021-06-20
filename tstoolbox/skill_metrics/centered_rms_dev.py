# -*- coding: utf-8 -*-
import numpy as np

from . import utils


def centered_rms_dev(predicted, reference):
    """
    Calculate the centered root-mean-square difference between two variables.

    The latter is calculated using the formula:

    (E')^2 = sum_(n=1)^N [(p_n - mean(p))(r_n - mean(r))]^2/N

    where p is the predicted values, r is the reference values, and
    N is the total number of values in p & r. Note that p & r must
    have the same number of values.

    Input:
    PREDICTED : predicted field
    REFERENCE : reference field

    Output:
    CRMSDIFF : centered root-mean-square (RMS) difference (E')^2

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com

    Created on Nov 24, 2016
    """
    # Check that dimensions of predicted and reference fields match
    utils.check_arrays(predicted, reference)

    # Calculate means
    pmean = np.mean(predicted)
    rmean = np.mean(reference)

    # Calculate (E')^2
    crmsd = np.square((predicted - pmean) - (reference - rmean))
    crmsd = np.sum(crmsd) / predicted.size
    crmsd = np.sqrt(crmsd)

    return crmsd
