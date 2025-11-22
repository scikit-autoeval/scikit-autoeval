.. -*- mode: rst -*-

|License| |Python| |Status|

.. |License| image:: https://img.shields.io/badge/license-BSD%203--Clause-blue.svg
   :target: https://opensource.org/licenses/BSD-3-Clause
.. |Python| image:: https://img.shields.io/pypi/pyversions/scikit-autoeval.svg
   :target: https://pypi.org/project/scikit-autoeval/
.. |Status| image:: https://img.shields.io/badge/status-beta-yellow.svg

.. image:: https://raw.githubusercontent.com/scikit-autoeval/scikit-autoeval/main/docs/_static/logo-dark.png
   :alt: scikit-autoeval logo
   :target: https://github.com/scikit-autoeval/scikit-autoeval

scikit-autoeval: Automatic Estimation of Model Performance Without Labels
=========================================================================

**scikit-autoeval** is an open-source Python library for *automatic evaluation* (AutoEval) of supervised machine learning models in scenarios where new labeled data is scarce or unavailable. Instead of relying only on traditional validation or test sets, it provides complementary estimators that infer how a trained model is likely to perform on unseen, unlabeled data domains.

Core Idea
---------
Most real-world deployments face some form of *data shift* or delayed labeling. scikit-autoeval helps bridge that gap by offering multiple independent signals of reliability—confidence filtering, statistical patterns of predictions, robustness under perturbations, cross-model agreement, and explanation-driven consistency—so practitioners can triangulate an estimated performance profile before costly annotation.

Implemented Estimators
----------------------
Below is a conceptual overview of the current estimation strategies. Each can handle multiple metrics (e.g. accuracy, F1) via a unified scorer interface.

1. **Confidence Threshold**: Counts only high-conviction predictions. Predictions below a confidence threshold are treated as likely errors when forming an expected outcome. Simple and fast; useful immediately in unlabeled settings.
2. **Regression-Based**: Learns a mapping between patterns in a model’s output probabilities (how concentrated or uncertain they are) and its true past performance across diverse datasets. Once learned, it can estimate performance of similar models on fresh unlabeled data.
3. **Regression + Noise**: Extends BR by training under controlled input perturbations to improve robustness under shift. Builds a richer meta-dataset by simulating degraded conditions.
4. **Agreement-Based**: Compares a primary model with a secondary alternative (e.g. a simpler classifier). Regions of disagreement are treated as warning zones; agreement suggests reliability.
5. **SHAP-Eval**: Uses model explanation values (SHAP) to train an auxiliary predictor of correctness, simulating expected labels by flipping predictions likely to be wrong. Repeats this process several times to smooth estimation.

All methods produce *estimated metrics*—they are heuristic and should be interpreted as indicative signals, not replacements for eventual ground-truth evaluation.

Quick Start
-----------
Install via pip:

.. code-block:: bash

   pip install scikit-autoeval

Basic usage with the confidence threshold evaluator:

.. code-block:: python

   import pandas as pd
   from sklearn.pipeline import make_pipeline
   from sklearn.impute import KNNImputer
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.metrics import accuracy_score, f1_score

   from skeval.evaluators.confidence import ConfidenceThresholdEvaluator
   from skeval.utils import get_cv_and_real_scores, print_comparison

   # Load two related medical datasets (source & target)
   src = pd.read_csv("./skeval/datasets/geriatria-controle-alzheimerLabel.csv")
   tgt = pd.read_csv("./skeval/datasets/neurologia-controle-alzheimerLabel.csv")

   Xs, ys = src.drop(columns=["Alzheimer"]), src["Alzheimer"]
   Xt, yt = tgt.drop(columns=["Alzheimer"]), tgt["Alzheimer"]

   model = make_pipeline(KNNImputer(n_neighbors=4),
                         RandomForestClassifier(n_estimators=300, random_state=42))

   scorers = {
       "accuracy": accuracy_score,
       "f1_macro": lambda y, p: f1_score(y, p, average="macro"),
   }

   evaluator = ConfidenceThresholdEvaluator(model=model, scorer=scorers)
   evaluator.fit(Xs, ys)
   estimated = evaluator.estimate(Xt, threshold=0.65)

   scores_dict = get_cv_and_real_scores(model=model,
                                        scorers=scorers,
                                        train_data=(Xs, ys),
                                        test_data=(Xt, yt))
   print_comparison(scorers,
                    scores_dict["cv_scores"],
                    estimated,
                    scores_dict["real_scores"])

Dependencies
------------
Minimum versions (see ``pyproject.toml`` for full list):

- Python >= 3.8
- scikit-learn >= 1.0
- numpy
- pandas
- matplotlib (plots & examples)
- shap (for SHAP-based estimator)
- xgboost (default inner classifier in SHAP-Eval)

Documentation
-------------
Online docs: https://scikit-autoeval.github.io/scikit-autoeval/

The documentation includes API references, evaluator examples, and conceptual guidance for interpreting estimates.

Citing
------
If you use scikit-autoeval in academic work, please cite the underlying techniques referenced in the estimators along with the library itself.

Example key references:

.. code-block:: text

   Guillory et al. (2021) – Predicting accuracy under shift.
   Deng et al. (2021) – Learning performance from unlabeled data.
   Madani et al. (2004) – Agreement/co-training perspectives.
   Silva & Veloso (2022) – SHAP-driven automatic evaluation.

Contributing
------------
Contributions are welcome—bug reports, feature ideas, estimator variants, documentation improvements. Please open an issue or pull request at:

- Source: https://github.com/scikit-autoeval/scikit-autoeval
- Issues: https://github.com/scikit-autoeval/scikit-autoeval/issues

To run tests locally:

.. code-block:: bash

   pip install -e .[dev]
   pytest -q

License
-------
Distributed under the BSD 3-Clause License.

Disclaimer
----------
Estimated scores should be treated as advisory signals. They complement, not replace, validation on labeled data once available.
