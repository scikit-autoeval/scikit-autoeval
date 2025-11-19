# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause

import unittest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Assuming the BaseEvaluator is in this path
from skeval.base import BaseEvaluator


class TestBaseEvaluator(unittest.TestCase):

    def setUp(self):
        # Basic model and scorers for use in tests
        self.model = LogisticRegression()
        self.scorers_dict = {"accuracy": accuracy_score, "f1_macro": f1_score}
        self.X_dummy = [[0], [1]]
        self.y_dummy = [0, 1]

    def test_initialization_and_attributes(self):
        """Test that attributes are set correctly during __init__."""
        evaluator = BaseEvaluator(
            model=self.model, scorer=self.scorers_dict, verbose=True
        )
        self.assertIs(evaluator.model, self.model)
        self.assertIs(evaluator.scorer, self.scorers_dict)
        self.assertTrue(evaluator.verbose)

    def test_default_parameters(self):
        """Test that default parameters are set correctly."""
        evaluator = BaseEvaluator(model=self.model)
        self.assertIs(evaluator.scorer, accuracy_score)
        self.assertFalse(evaluator.verbose)

    def test_fit_method_returns_self(self):
        """Test that the base fit() method (placeholder) returns 'self'."""
        evaluator = BaseEvaluator(model=self.model)
        # The base fit() does nothing but should return self for chaining
        instance = evaluator.fit(self.X_dummy, self.y_dummy)
        self.assertIs(instance, evaluator)

    def test_estimate_method_returns_none(self):
        """Test that the base estimate() method (placeholder) returns None."""
        evaluator = BaseEvaluator(model=self.model)
        # The base estimate() is just 'pass', so it returns None
        result = evaluator.estimate(self.X_dummy)
        self.assertIsNone(result)

    def test_get_scorer_names_with_dict(self):
        """Test _get_scorer_names when scorer is a dictionary."""
        evaluator = BaseEvaluator(model=self.model, scorer=self.scorers_dict)
        names = evaluator._get_scorer_names()
        # Use set for order-agnostic comparison
        self.assertEqual(set(names), {"accuracy", "f1_macro"})

    def test_get_scorer_names_with_callable(self):
        """Test _get_scorer_names when scorer is a single callable."""
        evaluator = BaseEvaluator(model=self.model, scorer=f1_score)
        names = evaluator._get_scorer_names()
        self.assertEqual(names, ["score"])

    def test_get_scorer_names_with_default(self):
        """Test _get_scorer_names with the default (accuracy_score)."""
        evaluator = BaseEvaluator(model=self.model)
        names = evaluator._get_scorer_names()
        self.assertEqual(names, ["score"])

    def test_get_scorer_names_with_invalid(self):
        """Test _get_scorer_names when scorer is not a dict or callable."""
        evaluator_none = BaseEvaluator(model=self.model, scorer=None)
        self.assertEqual(evaluator_none._get_scorer_names(), [])

        evaluator_str = BaseEvaluator(model=self.model, scorer="accuracy")
        self.assertEqual(evaluator_str._get_scorer_names(), [])

    def test_sklearn_get_params(self):
        """Test BaseEstimator get_params compatibility."""
        evaluator = BaseEvaluator(
            model=self.model, scorer=self.scorers_dict, verbose=True
        )
        params = evaluator.get_params(deep=False)
        expected_params = {
            "model": self.model,
            "scorer": self.scorers_dict,
            "verbose": True,
        }
        self.assertEqual(params, expected_params)

    def test_sklearn_set_params(self):
        """Test BaseEstimator set_params compatibility."""
        evaluator = BaseEvaluator(model=self.model, verbose=False)

        new_model = LogisticRegression(C=0.5)
        new_scorer = f1_score

        evaluator.set_params(model=new_model, scorer=new_scorer, verbose=True)

        self.assertTrue(evaluator.verbose)
        self.assertIs(evaluator.model, new_model)
        self.assertIs(evaluator.scorer, new_scorer)


if __name__ == "__main__":
    unittest.main()
