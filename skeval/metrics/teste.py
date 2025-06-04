from sklearn.metrics import mean_absolute_error

estimated_scores = {'acc': 0.8, 'f1': 0.8125}
real_scores = {'acc': 0.7, 'f1': 0.85}

print ({
    metric: mean_absolute_error([real_scores[metric]], [estimated_scores[metric]])
    for metric in real_scores
    if metric in estimated_scores
})
