from sklearn.metrics import accuracy_score, mean_absolute_error

def score_error(real_scores, estimated_scores, comparator=mean_absolute_error, verbose=False):
    """
    Compares estimated and real scores using a user-defined comparison function.

    Parameters:
    -----------
    real_scores : dict
        Scores computed with true labels.
    estimated_scores : dict
        Estimated scores without true labels.
    comparator : callable | dict
        Functions to compare (real, estimated). Default is mean_absolute_error.

    Returns:
    --------
    dict
        Dictionary with comparison results for each common scorer.
    """
    result = {}

    if callable(comparator):
        for metric in real_scores:
            if metric in estimated_scores:
                error = comparator([real_scores[metric]], [estimated_scores[metric]])
                result[metric] = error
                if verbose:
                    print(f"[{metric}] Real: {real_scores[metric]}, Estimated: {estimated_scores[metric]}, Error: {error}")
                    
    elif isinstance(comparator, dict):
        for metric in real_scores:
            if metric in estimated_scores and metric in comparator:
                error = comparator[metric]([real_scores[metric]], [estimated_scores[metric]])
                result[metric] = error
                if verbose:
                    print(f"[{metric}] Real: {real_scores[metric]}, Estimated: {estimated_scores[metric]}, Error: {error}")
    else:
        raise ValueError("Comparator must be a callable or a dict of callables.")

    return result
    
# def compare_real_estimated_cross_accuracy(train_X, train_y, test_X, test_y, evaluator: ConfidenceThresholdEvaluator):
#     evaluator.fit(train_X, train_y)
#     y_pred_real = evaluator.estimator.predict(test_X)
#     real_acc = accuracy_score(test_y, y_pred_real)

#     est_result = evaluator.estimate(test_X)
#     estimated_acc = list(est_result.values())[0] if est_result else 0.0

#     evaluator.fit(test_X, test_y)
#     y_pred_cross = evaluator.estimator.predict(train_X)
#     cross_acc = accuracy_score(train_y, y_pred_cross)
#     if evaluator.verbose:
#         print(f"Real Accuracy:     {real_acc:.4f}")
#         print(f"Estimated Accuracy:{estimated_acc:.4f}")
#         print(f"Cross Accuracy:    {cross_acc:.4f}")
#         print("-" * 50)

#     return {
#         "real_acc": real_acc,
#         "estimated_acc": estimated_acc,
#         "cross_acc": cross_acc
#     }