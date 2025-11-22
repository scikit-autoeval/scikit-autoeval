# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause


# Unified comparison plot for all evaluators
import os
import numpy as np
import matplotlib.pyplot as plt
from skeval.examples.confidence_example import run_confidence_eval
from skeval.examples.regression_example import run_regression_eval
from skeval.examples.regression_noise_example import run_regression_noise_eval
from skeval.examples.agreement_example import run_agreement_eval
from skeval.examples.shap_example import run_shap_eval

def get_main_metric(results, metric_name="accuracy"):
    # Fallback to first metric if not found
    if metric_name in results["real"]:
        return (
            results["cv"][metric_name],
            results["estimated"][metric_name],
            results["real"][metric_name],
        )
    else:
        m = list(results["real"].keys())[0]
        return (
            results["cv"][m], results["estimated"][m], results["real"][m]
        )

def plot_all_evaluators(metric_name="accuracy"):
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "plots", "graphics")
    )
    save_dir = os.path.normpath(base_dir)
    os.makedirs(save_dir, exist_ok=True)

    evaluators = [
        (run_confidence_eval, "Confidence Threshold"),
        (run_regression_eval, "Regression"),
        (run_regression_noise_eval, "Regression Noise"),
        (run_agreement_eval, "Agreement"),
        (run_shap_eval, "SHAP"),
    ]

    cv_scores, est_scores, real_scores, labels = [], [], [], []
    for func, label in evaluators:
        results = func(verbose=False)
        cv, est, real = get_main_metric(results, metric_name)
        cv_scores.append(cv)
        est_scores.append(est)
        real_scores.append(real)
        labels.append(label)

    x = np.arange(len(labels))
    width = 0.25

    plt.figure(figsize=(12, 6))
    plt.bar(x - width, cv_scores, width, label="via Cross-validation", color="#d6776b")
    plt.bar(x, est_scores, width, label="Estimada", color="#36579e")
    plt.bar(x + width, real_scores, width, label="Real", color="#21b564")

    # Título maior
    # plt.title(f"Comparação de ({metric_name}) - Todos os Avaliadores", fontsize=16)
    # Rótulo do eixo Y maior
    if metric_name == "accuracy":
        plt.ylabel("Acurácia", fontsize=14)
    else:
        plt.ylabel("F1-Macro", fontsize=14)
    # Rótulos do eixo X maiores
    plt.xticks(x, labels, rotation=20, fontsize=13)
    plt.ylim(0, 1.0)
    # Legenda maior
    plt.legend(fontsize=13)
    plt.grid(axis="y", linestyle="--", alpha=0.6)

    # Textos dos valores maiores
    for i, v in enumerate(cv_scores):
        plt.text(x[i] - width, v + 0.02, f"{v:.3f}", ha="center", fontsize=12)
    for i, v in enumerate(est_scores):
        plt.text(x[i], v + 0.02, f"{v:.3f}", ha="center", fontsize=12)
    for i, v in enumerate(real_scores):
        plt.text(x[i] + width, v + 0.02, f"{v:.3f}", ha="center", fontsize=12)

    plt.tight_layout()
    file_path = os.path.join(save_dir, f"all_evaluators_comparison_{metric_name}.png")
    plt.savefig(file_path, dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    plot_all_evaluators(metric_name="accuracy")
    plot_all_evaluators(metric_name="f1_macro")
