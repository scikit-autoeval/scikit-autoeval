# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any, Callable, Dict, Mapping, Union
from sklearn.metrics import mean_absolute_error


def score_error(
    real_scores: Mapping[str, float],
    est_scores: Mapping[str, float],
    comparator: Union[
        Callable[[Any, Any], float], Mapping[str, Callable[[Any, Any], float]]
    ] = mean_absolute_error,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Compares estimated and real scores using a user-defined comparison function.

    This function iterates through the metrics present in both `real_scores` and
    `est_scores` dictionaries and computes the error between them using
    the provided comparator function(s).

    Parameters
    ----------
    real_scores : dict
        A dictionary of scores computed with true labels.
        Example: `{'accuracy': 0.9, 'f1': 0.85}`
    est_scores : dict
        A dictionary of scores estimated without true labels.
        Example: `{'accuracy': 0.88, 'f1': 0.82}`
    comparator : callable or dict, default=mean_absolute_error
        The function or dictionary of functions used to compare the real and
        estimated scores.
        - If callable, it's applied to all common metrics.
        - If dict, it maps a metric name to a specific comparator function.
    verbose : bool, default=False
        If True, prints the real score, estimated score, and the resulting
        error for each metric.

    Returns
    -------
    dict
        A dictionary containing the comparison results (errors) for each
        common metric.

    Raises
    ------
    ValueError
        If `comparator` is not a callable or a dictionary of callables.

    Examples
    --------
    >>> real = {'accuracy': 0.95, 'precision': 0.90, 'recall': 0.85}
    >>> estimated = {'accuracy': 0.91, 'precision': 0.92, 'f1_score': 0.88}

    >>> # Example 1: Using the default comparator (mean_absolute_error)
    >>> errors = score_error(real, estimated)
    >>> for metric, error in sorted(errors.items()):
    ...     print(f"{metric}: {error:.4f}")
    accuracy: 0.0400
    precision: 0.0200

    >>> # Example 2: Using a dictionary of different comparators
    >>> from sklearn.metrics import mean_squared_error
    >>> comparators = {
    ...     'accuracy': mean_absolute_error,
    ...     'precision': mean_squared_error
    ... }
    >>> errors_custom = score_error(real, estimated, comparator=comparators, verbose=True)
    [accuracy] Real: 0.95, Estimated: 0.91, Error: 0.040000000000000036
    [precision] Real: 0.9, Estimated: 0.92, Error: 0.0004000000000000003
    >>> for metric, error in sorted(errors_custom.items()):
    ...     print(f"{metric}: {error:.4f}")
    accuracy: 0.0400
    precision: 0.0004
    """
    result = {}

    if callable(comparator):
        for metric in real_scores:
            if metric in est_scores:
                error = comparator([real_scores[metric]], [est_scores[metric]])
                result[metric] = error
                if verbose:
                    print(
                        f"[{metric}] Real: {real_scores[metric]}, " +
                        f"Estimated: {est_scores[metric]}, " +
                        f"Error: {error}"
                    )

    elif isinstance(comparator, dict):
        for metric in real_scores:
            if metric in est_scores and metric in comparator:
                error = comparator[metric](
                    [real_scores[metric]], [est_scores[metric]]
                )
                result[metric] = error
                if verbose:
                    print(
                        f"[{metric}] Real: {real_scores[metric]}, " +
                        f"Estimated: {est_scores[metric]}, " +
                        f"Error: {error}"
                    )
    else:
        raise ValueError("Comparator must be a callable or a dict of callables.")

    return result
