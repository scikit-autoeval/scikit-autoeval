# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.exceptions import NotFittedError

# Assuming the ShapEvaluator is in this path
from skeval.evaluators.shap import ShapEvaluator

class TestShapEvaluator(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load a binary classification dataset
        X, y = load_breast_cancer(return_X_y=True)

        # Split data. ShapEvaluator needs x_train for fit() and X_eval for estimate()
        cls.x_train, cls.X_eval, cls.y_train, cls.y_eval = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # A small number of predictions for fast tests, due to the random loop
        cls.n_pred_fast = 5

    def test_fit_and_estimate_with_single_scorer(self):
        """
        Test fit and estimate with a single scorer.
        Requires a tree-based model for shap.TreeExplainer.
        """
        # RandomForest is a tree-based model
        model = RandomForestClassifier(n_estimators=10, random_state=42)

        evaluator = ShapEvaluator(model=model, scorer=accuracy_score, verbose=False)

        # Fit the model and set x_train/y_train inside the evaluator
        evaluator.fit(self.x_train, self.y_train)

        # Estimate on the evaluation set
        estimated_scores = evaluator.estimate(self.X_eval, n_pred=self.n_pred_fast)

        self.assertIsInstance(estimated_scores, dict)
        self.assertIn("score", estimated_scores)
        self.assertIsInstance(estimated_scores["score"], float)
        self.assertGreaterEqual(estimated_scores["score"], 0.0)
        self.assertLessEqual(estimated_scores["score"], 1.0)

    def test_fit_and_estimate_with_multiple_scorers(self):
        """
        Test fit and estimate with multiple scorers.
        """
        # XGBClassifier is also a tree-based model
        model = XGBClassifier(n_estimators=10, random_state=42, eval_metric="logloss")

        scorers = {
            "accuracy": accuracy_score,
            "f1_macro": lambda y_true, y_pred: f1_score(
                y_true, y_pred, average="macro"
            ),
        }

        evaluator = ShapEvaluator(model=model, scorer=scorers, verbose=False)

        evaluator.fit(self.x_train, self.y_train)

        estimated_scores = evaluator.estimate(self.X_eval, n_pred=self.n_pred_fast)

        self.assertIsInstance(estimated_scores, dict)
        self.assertIn("accuracy", estimated_scores)
        self.assertIn("f1_macro", estimated_scores)
        for v in estimated_scores.values():
            self.assertIsInstance(v, float)

    def test_estimate_without_fit_raises_error(self):
        """
        Test that estimate() raises NotFittedError if model is not fitted.
        """
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        evaluator = ShapEvaluator(model=model)

        # This checks the check_is_fitted(self.model) call
        with self.assertRaises(NotFittedError):
            evaluator.estimate(self.X_eval)

    def test_estimate_without_train_data_raises_error(self):
        """
        Test that estimate() raises ValueError if x_train/y_train are missing,
        even if the model is pre-fitted.
        """
        # Fit the model *outside* the evaluator
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.x_train, self.y_train)

        # Init evaluator with pre-fit model, but *without* x_train/y_train
        evaluator = ShapEvaluator(model=model)

        # This checks the "if self.x_train is None..." call in estimate()
        with self.assertRaises(ValueError):
            evaluator.estimate(self.X_eval)

    def test_init_with_prefit_model_and_train_data(self):
        """
        Test that estimate() works without calling fit() if a pre-fitted model
        and training data are provided to __init__.
        """
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(self.x_train, self.y_train)

        # Provide all required components to __init__
        evaluator = ShapEvaluator(
            model=model, x_train=self.x_train, y_train=self.y_train
        )

        # Should work without calling evaluator.fit()
        try:
            estimated_scores = evaluator.estimate(self.X_eval, n_pred=self.n_pred_fast)
            self.assertIn("score", estimated_scores)
            self.assertIsInstance(estimated_scores["score"], float)
        except (NotFittedError, ValueError):
            self.fail("estimate() failed even with pre-fit model and train data")

    def test_default_inner_clf_is_xgb(self):
        """
        Test that the default inner_clf is XGBClassifier.
        """
        model = RandomForestClassifier()
        evaluator = ShapEvaluator(model=model)

        self.assertIsInstance(evaluator.inner_clf, XGBClassifier)


if __name__ == "__main__":
    unittest.main()
