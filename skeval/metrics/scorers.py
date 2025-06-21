# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
def make_scorer(func, **kwargs):
    """
    Wraps a metric function with fixed keyword arguments into a simple scorer.

    This utility is useful for creating a unified scorer interface from metric
    functions that require specific arguments (like `average='macro'` for f1_score).

    Parameters
    ----------
    func : callable
        A metric function from a library like scikit-learn, such as
        `accuracy_score`, `f1_score`, etc.
    **kwargs : dict
        Keyword arguments to be permanently passed to the metric function
        whenever the scorer is called.

    Returns
    -------
    callable
        A new scorer function that accepts only `y_true` and `y_pred` as arguments.

    Examples
    --------
    >>> from sklearn.metrics import f1_score
    >>> import numpy as np

    >>> # Ground truth and predictions for a multi-class problem
    >>> y_true = np.array([0, 1, 2, 0, 1, 2])
    >>> y_pred = np.array([0, 2, 1, 0, 0, 1])

    >>> # Create a scorer for F1-score with 'macro' averaging
    >>> macro_f1_scorer = make_scorer(f1_score, average='macro')

    >>> # Use the new scorer
    >>> score = macro_f1_scorer(y_true, y_pred)
    >>> print(f"Macro F1 Score: {score:.4f}")
    Macro F1 Score: 0.2667

    >>> # The result is identical to calling f1_score directly with the argument
    >>> direct_call_score = f1_score(y_true, y_pred, average='macro')
    >>> print(f"Direct call F1 Score: {direct_call_score:.4f}")
    Direct call F1 Score: 0.2667
    >>> np.isclose(score, direct_call_score)
    True
    """
    def scorer(y_true, y_pred):
        return func(y_true, y_pred, **kwargs)
    return scorer