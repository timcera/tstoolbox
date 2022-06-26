# -*- coding: utf-8 -*-
import numpy as np

from . import utils


def rmsd(predicted, reference):
    """
    Calculate root-mean-square deviation (RMSD) between two variables.

    Calculates the root-mean-square deviation between two variables
    PREDICTED and REFERENCE. The RMSD is calculated using the
    formula:

    RMSD^2 = sum_(n=1)^N [(p_n - r_n)^2]/N

    where p is the predicted values, r is the reference values, and
    N is the total number of values in p & r. Note that p & r must
    have the same number of values.

    Input:
    PREDICTED : predicted values
    REFERENCE : reference values

    Output:
    R : root-mean-square deviation (RMSD)

    Author: Peter A. Rochford
        Symplectic, LLC
        www.thesymplectic.com
        prochford@thesymplectic.com

    Created on Dec 9, 2016
    """
    # Check that dimensions of predicted and reference fields match
    utils.check_arrays(predicted, reference)

    return np.sqrt(np.sum(np.square(predicted - reference)) / len(predicted))
