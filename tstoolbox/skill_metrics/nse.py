import numpy as np

from . import utils


def nse(simulated, observed):
    """
    Calculate the Nash-Sutcliffe efficiency.

    Calculates the Nash-Sutcliffe efficiency between two variables
    simulated and observed. The NSE is calculated using the
    formula:

    NSE = 1 - sum_(n=1)^N (s_n - o_n)^2 / sum_(n=1)^N (o_n - mean(r))^2

    where s is the simulated values, o is the observed values, and
    N is the total number of values in s & o. Note that s & o must
    have the same number of values.

    Nash-Sutcliffe efficiency can range from -infinity to 1. An efficiency of
    1 (E = 1) corresponds to a perfect match of modeled discharge to the
    observed data. An efficiency of 0 (E = 0) indicates that the model
    predictions are as accurate as the mean of the observed data, whereas an
    efficiency less than zero (E < 0) occurs when the observed mean is a better
    predictor than the model or, in other words, when the residual variance
    (described by the numerator in the expression above), is larger than the
    data variance (described by the denominator). Essentially, the closer the
    model efficiency is to 1, the more accurate the model is.

    The efficiency coefficient is sensitive to extreme values and might yield
    sub-optimal results when the dataset contains large outliers in it.

    Nash-Sutcliffe efficiency can be used to quantitatively describe the
    accuracy of model outputs other than discharge. This method can be used to
    describe the predictive accuracy of other models as long as there is
    observed data to compare the model results to. For example, Nash-Sutcliffe
    efficiency has been reported in scientific literature for model simulations
    of discharge, and water quality constituents such as sediment, nitrogen,
    and phosphorus loading.

    Input:
    simulated : simulated values
    observed : observed values

    Output:
    NSE : Nash-Sutcliffe Efficiency
    """
    # Check that dimensions of simulated and observed fields match
    utils.check_arrays(simulated, observed)

    # Calculate the NSE
    nse = 1 - (np.sum((simulated - observed)**2) /
               np.sum((observed - np.mean(observed))**2))

    return nse
