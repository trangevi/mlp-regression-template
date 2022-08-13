"""
This module defines the following routines used by the 'train' step of the regression pipeline:

- ``estimator_fn``: Defines the customizable estimator type and parameters that are used
  during training to produce a model pipeline.
"""


def estimator_fn():
    """
    Returns an *unfitted* estimator that defines ``fit()`` and ``predict()`` methods.
    The estimator's input and output signatures should be compatible with scikit-learn
    estimators.
    """
    """
    from sklearn.linear_model import SGDRegressor

    return SGDRegressor(random_state=42)
    """

    from sklearn import linear_model

    alpha = .5
    l1_ratio = 1
    max_iter = 50
    regr = linear_model.ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter, random_state=0)

    return regr
