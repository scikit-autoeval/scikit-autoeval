scikit-autoeval: Documentation and User Guide
=============================================

**scikit-autoeval** is an open-source Python library designed to automate the evaluation of supervised machine learning models.  
It provides standardized interfaces and metrics for model validation, ensuring compatibility with the Scikit-learn ecosystem.  
The library simplifies the benchmarking process and supports the development of robust and reproducible experiments.

Overview
--------

This section gives a concise description of the library's purpose, how the
modules connect, and how to use it in a typical workflow.

Purpose
^^^^^^^

- **Goal**: estimate, compare, and validate the performance of classification
   models when labeled target-domain data is limited or unavailable. The
   library provides evaluators that produce performance estimates using
   signals derived from the model itself (e.g., confidence, agreement between
   models, SHAP values, or meta-regression with label noise).

Module structure and relationships
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

- `skeval.evaluators`: contains the primary evaluators
   (`ConfidenceThresholdEvaluator`, `RegressionEvaluator`,
   `RegressionNoiseEvaluator`, `AgreementEvaluator`, `ShapEvaluator`). Each
   evaluator encapsulates a strategy for estimating metrics without target
   labels and follows a common pattern: `fit()` to prepare the evaluator and
   `estimate()` to apply it on the target dataset.
- `skeval.metrics`: utility functions and scorer wrappers that standardize
   metric calculation and comparison.
- `skeval.utils`: helper utilities such as `get_cv_and_real_scores` and
   `print_comparison`, used in examples and for validating evaluator output.
- `skeval.examples`: example scripts showing end-to-end flows for each
   evaluator; useful as references or starting points for integration.

Typical usage flow
^^^^^^^^^^^^^^^^^^

1. Prepare one or more source-domain training datasets. Some evaluators
    (e.g., `RegressionEvaluator` and `RegressionNoiseEvaluator`) require
    multiple datasets to train meta-models.
2. Define a preprocessing + classifier pipeline (for example
    `sklearn.pipeline.make_pipeline(KNNImputer(...), RandomForestClassifier(...))`).
3. Initialize the desired evaluator passing `model` and `scorers`.
4. Call `fit(X_list, y_list)` when applicable to train meta-regressors or
    other internal components.
5. Use `estimate(X_unlabeled)` on the target data to get performance
    estimates. When target labels are available, compare estimates with
    `get_cv_and_real_scores(...)` to validate results.

Why use this library
^^^^^^^^^^^^^^^^^^^^

- Automates cross-domain evaluation strategies that would otherwise be
   repetitive to implement.
- Provides a consistent API across multiple techniques (same `fit`/`estimate`
   methods), simplifying experimentation and comparison.
- Includes ready-made utilities to compute CV/real scores and to compare
   results reproducibly.

Installation
------------

To install the library from PyPI, run:
.. code-block:: bash

   pip install scikit-autoeval

Quick Example
-------------
A short runnable example showing how to use the `ShapEvaluator` to estimate performance and compare it to cross-validated and real scores:

.. code-block:: python

      import pandas as pd
      from sklearn.metrics import accuracy_score, f1_score
      from sklearn.impute import KNNImputer
      from sklearn.pipeline import make_pipeline
      from xgboost import XGBClassifier

      from skeval.evaluators.shap import ShapEvaluator
      from skeval.utils import get_cv_and_real_scores, print_comparison


      # 1. Load datasets
      geriatrics = pd.read_csv("./skeval/datasets/geriatria-controle-alzheimerLabel.csv")
      neurology = pd.read_csv("./skeval/datasets/neurologia-controle-alzheimerLabel.csv")

      # 2. Separate features and target
      X1, y1 = geriatrics.drop(columns=["Alzheimer"]), geriatrics["Alzheimer"]
      X2, y2 = neurology.drop(columns=["Alzheimer"]), neurology["Alzheimer"]

      # 3. Define pipeline (KNNImputer + XGBoost)
      model = make_pipeline(KNNImputer(n_neighbors=5), XGBClassifier())

      # 4. Define scorers and evaluator
      scorers = {
         "accuracy": accuracy_score,
         "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
      }

      evaluator = ShapEvaluator(
         model=model,
         scorer=scorers,
         verbose=False,
         inner_clf=XGBClassifier(random_state=42),
      )

      # 5. Fit evaluator on geriatrics data
      evaluator.fit(X1, y1)

      # 6. Estimate performance (train on X1, estimate on X2)
      estimated_scores = evaluator.estimate(X2)

      # 7. Compute real and CV performance
      train_data = X1, y1
      test_data = X2, y2
      scores_dict = get_cv_and_real_scores(
         model=model, scorers=scorers, train_data=train_data, test_data=test_data
      )
      cv_scores = scores_dict["cv_scores"]
      real_scores = scores_dict["real_scores"]

      print_comparison(scorers, cv_scores, estimated_scores, real_scores)

Modules
-------

.. toctree::
   :maxdepth: 2

   modules

* :ref:`modindex`
