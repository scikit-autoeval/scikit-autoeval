# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

import os
import numpy as np
import matplotlib.pyplot as plt
from ..exemples.confidence_exemple import run_confidence_eval
from ..exemples.regression_exemple import run_regression_eval
from ..exemples.regression_noise_exemple import run_regression_noise_eval
from ..exemples.shap_exemple import run_shap_eval

def plot_eval(eval, eval_name="", file_name=""):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "plots", "graphics"))
    save_dir = os.path.normpath(base_dir)

    os.makedirs(save_dir, exist_ok=True)

    results = eval(verbose=False)
    metrics = list(results['real'].keys())

    cv_values = [results['cv'][m] for m in metrics]
    est_values = [results['estimated'][m] for m in metrics]
    real_values = [results['real'][m] for m in metrics]

    x = np.arange(len(metrics))
    width = 0.25

    plt.figure(figsize=(8, 5))
    plt.bar(x - width, cv_values, width, label='CV', color='#6baed6')
    plt.bar(x, est_values, width, label='Estimated', color='#9ecae1')
    plt.bar(x + width, real_values, width, label='Real', color='#2171b5')

    plt.title(eval_name + " - Performance Comparison")
    plt.ylabel("Score")
    plt.xticks(x, metrics)
    plt.ylim(0, 1.0)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    for i, v in enumerate(cv_values):
        plt.text(x[i] - width, v + 0.02, f"{v:.3f}", ha='center', fontsize=8)
    for i, v in enumerate(est_values):
        plt.text(x[i], v + 0.02, f"{v:.3f}", ha='center', fontsize=8)
    for i, v in enumerate(real_values):
        plt.text(x[i] + width, v + 0.02, f"{v:.3f}", ha='center', fontsize=8)

    plt.tight_layout()

    file_path = os.path.join(save_dir, file_name+"_comparison.png")
    plt.savefig(file_path, dpi=300, bbox_inches='tight')

plot_eval(run_confidence_eval, eval_name="Confidence Threshold Evaluator", file_name="confidence_eval")
plot_eval(run_regression_eval, eval_name="Regression Evaluator", file_name="regression_eval")
plot_eval(run_regression_noise_eval, eval_name="Regression Noise Evaluator", file_name="regression_noise_eval")
plot_eval(run_shap_eval, eval_name="SHAP Evaluator", file_name="shap_eval")