# -*- coding: utf-8 -*-
"""Collection of functions for the manipulation of time series."""

from __future__ import absolute_import, division, print_function

import warnings
from typing import List, Optional, Union

import mando
import pandas as pd
import typic
from mando.rst_text_formatter import RSTHelpFormatter
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score

from .. import tsutils

warnings.filterwarnings("ignore")

_FUNCS = {
    "ARD": linear_model.ARDRegression,
    "BayesianRidge": linear_model.BayesianRidge,
    "ElasticNetCV": linear_model.ElasticNetCV,
    "ElasticNet": linear_model.ElasticNet,
    "Huber": linear_model.HuberRegressor,
    "LarsCV": linear_model.LarsCV,
    "Lars": linear_model.Lars,
    "LassoCV": linear_model.LassoCV,
    "LassoLarsCV": linear_model.LassoLarsCV,
    "LassoLarsIC": linear_model.LassoLarsIC,
    "LassoLars": linear_model.LassoLars,
    "Lasso": linear_model.Lasso,
    "Linear": linear_model.LinearRegression,
    #    "LogisticCV": linear_model.LogisticRegressionCV,
    #    "Logistic": linear_model.LogisticRegression,
    #    "MultiTaskElasticNetCV": linear_model.MultiTaskElasticNetCV,
    #    "MultiTaskElasticNet": linear_model.MultiTaskElasticNet,
    #    "MultiTaskLassoCV": linear_model.MultiTaskLassoCV,
    #    "MultiTaskLasso": linear_model.MultiTaskLasso,
    "OrthogonalMatchingPursuitCV": linear_model.OrthogonalMatchingPursuitCV,
    "OrthogonalMatchingPursuit": linear_model.OrthogonalMatchingPursuit,
    #    "PassiveAggressive": linear_model.PassiveAggressiveClassifier,
    #    "Perceptron": linear_model.Perceptron,
    "RANSAC": linear_model.RANSACRegressor,
    "RidgeCV": linear_model.RidgeCV,
    "Ridge": linear_model.Ridge,
    "SGD": linear_model.SGDRegressor,
    "TheilSen": linear_model.TheilSenRegressor,
}


@mando.command("regression", formatter_class=RSTHelpFormatter, doctype="numpy")
@tsutils.doc(tsutils.docstrings)
def regression_cli(
    method,
    x_train_cols,
    y_train_col,
    x_pred_cols=None,
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    print_input=False,
    tablefmt="csv",
    por=False,
):
    """Regression of one or more time-series or indices to a time-series.

    If optional ``x_pred_cols`` is given will return a time-series of the ``y``
    predictions. Otherwise returns dictionary of equation and statistics about
    the regression fit.

    Parameters
    ----------
    method: str
        The method of regression.  The chosen method will use `x_train_cols` as the
        independent data and `y_pred_col` as the dependent data.

        ARD
            Requires lots of memory.

            Fit the weights of a regression model, using an ARD prior. The
            weights of the regression model are assumed to be in Gaussian
            distributions. Also estimate the parameters lambda (precisions of
            the distributions of the weights) and alpha (precision of the
            distribution of the noise). The estimation is done by an iterative
            procedures (Evidence Maximization)
        BayesianRidge
            Fit a Bayesian ridge model. See the Notes section for details on
            this implementation and the optimization of the regularization
            parameters lambda (precision of the weights) and alpha (precision
            of the noise).
        ElasticNetCV
            Elastic Net model with iterative fitting along a regularization
            path.
        ElasticNet
            Linear regression with combined L1 and L2 priors as regularizer.
        Huber
            Linear regression model that is robust to outliers.

            The Huber Regressor optimizes the squared loss for the samples
            where abs((y - X'w) / sigma) < epsilon and the absolute loss for
            the samples where abs((y - X'w) / sigma) > epsilon, where w and
            sigma are parameters to be optimized. The parameter sigma makes
            sure that if y is scaled up or down by a certain factor, one does
            not need to rescale epsilon to achieve the same robustness. Note
            that this does not take into account the fact that the different
            features of X may be of different scales.

            This makes sure that the loss function is not heavily influenced by
            the outliers while not completely ignoring their effect.
        LarsCV
            Cross-validated Least Angle Regression model.
        Lars
            Least Angle Regression model.
        LassoCV
            Lasso linear model with iterative fitting along a regularization
            path.
        LassoLarsCV
            Cross-validated Lasso, using the LARS algorithm.
        LassoLarsIC
            Lasso model fit with Lars using BIC or AIC for model selection.
        LassoLars
            Lasso model fit with Least Angle Regression a.k.a. Lars.  It is
            a Linear Model trained with an L1 prior as regularizer.
        Lasso
            Linear Model trained with L1 prior as regularizer (aka the Lasso).
        Linear
            LinearRegression fits a linear model with coefficients w = (w1, …,
            wp) to minimize the residual sum of squares between the observed
            targets in the dataset, and the targets predicted by the linear
            approximation.
        RANSAC
            RANSAC (RANdom SAmple Consensus) algorithm.  RANSAC is an iterative
            algorithm for the robust estimation of parameters from a subset of
            inliers from the complete data set.
        RidgeCV
            Ridge regression with built-in cross-validation.  By default, it
            performs Generalized Cross-Validation, which is a form of efficient
            Leave-One-Out cross-validation.
        Ridge
            This classifier first converts the target values into (-1, 1) and
            then treats the problem as a regression task (multi-output
            regression in the multiclass case).
        SGD
            Input must be scaled by removing mean and scaling to unit variance.
            Can use 'tstoolbox normalization ...' to scale the input.

            Linear model fitted by minimizing a regularized empirical loss with
            SGD.  SGD stands for Stochastic Gradient Descent: the gradient of
            the loss is estimated each sample at a time and the model is
            updated along the way with a decreasing strength schedule (aka
            learning rate).

            The regularizer is a penalty added to the loss function that
            shrinks model parameters towards the zero vector using either the
            squared euclidean norm L2 or the absolute norm L1 or a combination
            of both (Elastic Net). If the parameter update crosses the 0.0
            value because of the regularizer, the update is truncated to 0.0 to
            allow for learning sparse models and achieve online feature
            selection.
        TheilSen
            Theil-Sen Estimator: robust multivariate regression model.

            The algorithm calculates least square solutions on subsets with
            size n_subsamples of the samples in X. Any value of n_subsamples
            between the number of features and samples leads to an estimator
            with a compromise between robustness and efficiency. Since the
            number of least square solutions is “n_samples choose
            n_subsamples”, it can be extremely large and can therefore be
            limited with max_subpopulation. If this limit is reached, the
            subsets are chosen randomly. In a final step, the spatial median
            (or L1 median) is calculated of all least square solutions.
    x_train_cols: str or list
        List of column names/numbers that hold the ``x`` value datasets used to
        train the regression.  Perform a multiple regression if ``method``
        allows by giving several ``x_train_cols``.  To include the index in the
        regression use column 0 or the index name.
    y_train_col: str or list
        Column name or number of the ``y`` dataset used to train the
        regression.

        The ``y_train_col`` cannot be part of ``x_train_cols`` or
        ``x_pred_cols``.
    x_pred_cols : str or list
        [optional, if supplied will return a time-series of the ``y``
        prediction based on ``x_pred_cols``.]

        List of column names/numbers of ``x`` value datasets used to create the
        ``y`` prediction.  Needs to be the same number of columns as
        ``x_train_cols``.  Can be identical columns to ``x_train_cols``.
    ${input_ts}
    ${columns}
    ${start_date}
    ${end_date}
    ${dropna}
    ${clean}
    ${round_index}
    ${skiprows}
    ${index_type}
    ${names}
    ${print_input}
    ${tablefmt}
    ${por}
    """
    tsutils.printiso(
        regression(
            method,
            x_train_cols,
            y_train_col,
            x_pred_cols=x_pred_cols,
            input_ts=input_ts,
            columns=columns,
            start_date=start_date,
            end_date=end_date,
            dropna=dropna,
            clean=clean,
            round_index=round_index,
            skiprows=skiprows,
            index_type=index_type,
            names=names,
            print_input=print_input,
            por=por,
        ),
        tablefmt=tablefmt,
        headers=[],
    )


# @tsutils.validator(
#    method=[str, ["domain", _FUNCS.keys()], None],
# )
@tsutils.transform_args(
    x_train_cols=tsutils.make_list,
    y_train_col=tsutils.make_list,
    x_pred_cols=tsutils.make_list,
)
@typic.al
@tsutils.copy_doc(regression_cli)
def regression(
    method,
    x_train_cols: List[Union[str, int]],
    y_train_col: List[Union[str, int]],
    x_pred_cols: Optional[List[Union[str, int]]] = None,
    input_ts="-",
    columns=None,
    start_date=None,
    end_date=None,
    dropna="no",
    clean=False,
    round_index=None,
    skiprows=None,
    index_type="datetime",
    names=None,
    print_input=False,
    por=False,
):
    """Regression of data."""
    for to in y_train_col:
        for fro in x_train_cols:
            if to == fro:
                raise ValueError(
                    tsutils.error_wrapper(
                        f"""
You can't have columns in both "x_train_cols", and "y_train_col"
keywords.  Instead you have "{to}" in both.
"""
                    )
                )

    tsd = tsutils.common_kwds(
        input_ts,
        skiprows=skiprows,
        names=names,
        index_type=index_type,
        start_date=start_date,
        end_date=end_date,
        pick=columns,
        round_index=round_index,
        dropna=dropna,
        clean=clean,
        por=por,
    )

    if print_input is True:
        ntsd = tsd.copy()
    else:
        ntsd = tsd

    ntsd = tsutils.asbestfreq(ntsd)

    testfreqstr = ntsd.index.freqstr.lstrip("0123456789")
    if testfreqstr[0] == "A":
        ntsd[ntsd.index.name + "_"] = ntsd.index.year - ntsd.index[0].year
    elif testfreqstr[0] == "M":
        ntsd[ntsd.index.name + "_"] = (ntsd.index.year - ntsd.index[0].year) * 12 + (
            ntsd.index.month - ntsd.index[0].month
        )
    else:
        try:
            # In case ntsd.index.freqstr is a multiple, for example "15T".
            ntsd[ntsd.index.name + "_"] = (ntsd.index - ntsd.index[0]) // pd.Timedelta(
                ntsd.index.freqstr
            )
        except ValueError:
            ntsd[ntsd.index.name + "_"] = (ntsd.index - ntsd.index[0]) // pd.Timedelta(
                "1" + ntsd.index.freqstr
            )

    if x_pred_cols is None:
        nx_pred_cols = x_train_cols
    else:
        nx_pred_cols = x_pred_cols

    x_train_cols = tsutils.make_iloc(ntsd.columns, x_train_cols)
    y_train_col = tsutils.make_iloc(ntsd.columns, y_train_col)

    wtsd = ntsd.iloc[:, x_train_cols + y_train_col]

    # Train on 'any' dropna rows
    wtsddna = wtsd.dropna()
    # Train on last column
    y_train = wtsddna.iloc[:, -1].values
    # with all other columns
    x_train = wtsddna.iloc[:, :-1].values

    regr = _FUNCS[method]()
    regr.fit(x_train, y_train)

    if x_pred_cols is None:
        x_pred = x_train
    else:
        nx_pred_cols = tsutils.make_iloc(ntsd.columns, x_pred_cols)
        x_pred = ntsd.iloc[:, nx_pred_cols].dropna()
    y_pred = regr.predict(x_pred)

    if x_pred_cols is None:
        if method == "RANSAC":
            regr = regr.estimator_
        rdata = []
        rdata.append(["Coefficients", regr.coef_])
        rdata.append(["Intercept", regr.intercept_])
        rdata.append(["Mean squared error", mean_squared_error(y_train, y_pred)])
        rdata.append(["Coefficient of determination", r2_score(y_train, y_pred)])
        return rdata
    result = pd.DataFrame(y_pred, index=x_pred.index)
    result = result.reindex(index=wtsd.index)
    return tsutils.return_input(print_input, tsd, result)


if __name__ == "__init__":
    pass
