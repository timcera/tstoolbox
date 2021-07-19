# -*- coding: utf-8 -*-
import numpy as np

from . import utils


def kge(simulated, observed):
    """
    Calculate the Kling-Gupta efficiency.

    Calculates the Kling-Gupta efficiency between two variables
    simulated and observed. The kge is calculated using the
    formula:

    KGE = 1 - sqrt((cc-1)**2 + (alpha-1)**2 + (beta-1)**2)

    where:
        cc = correlation coefficient between simulated and observed;
        alpha = std(simulated) / std(observed)
        beta = sum(simulated) / sum(observed)

    where s is the simulated values, o is the observed values, and
    N is the total number of values in s & o. Note that s & o must
    have the same number of values.

    Kling-Gupta efficiency can range from -infinity to 1. An efficiency of 1 (E
    = 1) corresponds to a perfect match of model to observed data.
    Essentially, the closer the model efficiency is to 1, the more accurate the
    model is.

    The efficiency coefficient is sensitive to extreme values and might yield
    sub-optimal results when the dataset contains large outliers in it.

    Kling-Gupta efficiency can be used to quantitatively describe the accuracy
    of model outputs. This method can be used to describe the predictive
    accuracy of other models as long as there is observed data to compare the
    model results to.

    Input:
    simulated : simulated values
    observed : observed values

    Output:
    kge : Kling-Gupta Efficiency
    """
    # Check that dimensions of simulated and observed fields match
    utils.check_arrays(simulated, observed)

    alpha = np.std(simulated) / np.std(observed)
    beta = np.sum(simulated) / np.sum(observed)
    cc = np.corrcoef(observed, simulated)[0, 1]

    # Calculate the kge
    kge = 1.0 - np.sqrt((cc - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2)

    return kge
