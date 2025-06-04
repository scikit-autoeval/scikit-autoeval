def make_scorer(func, **kwargs):
    """
    Wraps a metric function with fixed keyword arguments into a simple scorer.

    Parameters:
    -----------
    func : callable
        A metric function like accuracy_score, f1_score, etc.
    kwargs : dict
        Keyword arguments to be passed to the metric function.

    Returns:
    --------
    callable
        A function that accepts y_true and y_pred.
    """
    def scorer(y_true, y_pred):
        return func(y_true, y_pred, **kwargs)
    return scorer
