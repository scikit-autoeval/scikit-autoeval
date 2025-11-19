# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
import unittest
import io
from contextlib import redirect_stdout
from sklearn.metrics import mean_absolute_error, mean_squared_error

from skeval.metrics.comparison import score_error


class TestScoreError(unittest.TestCase):

    def setUp(self):
        """Set up the test data to be used across multiple methods."""
        self.real_scores = {"accuracy": 0.95, "precision": 0.90, "recall": 0.85}
        self.estimated_scores = {"accuracy": 0.91, "precision": 0.92, "f1_score": 0.88}

    def test_default_comparator(self):
        """Test the function with the default comparator (mean_absolute_error)."""
        expected_errors = {
            "accuracy": 0.040000000000000036,  # abs(0.95 - 0.91)
            "precision": 0.020000000000000018,  # abs(0.90 - 0.92)
        }
        errors = score_error(self.real_scores, self.estimated_scores)
        self.assertEqual(errors.keys(), expected_errors.keys())
        for metric in errors:
            self.assertAlmostEqual(errors[metric], expected_errors[metric])

    def test_dict_of_comparators(self):
        """Test the function with a dictionary of different comparators."""
        comparators = {"accuracy": mean_absolute_error, "precision": mean_squared_error}
        expected_errors = {
            "accuracy": 0.040000000000000036,  # abs(0.95 - 0.91)
            "precision": 0.0004000000000000003,  # (0.90 - 0.92)^2
        }
        errors = score_error(
            self.real_scores, self.estimated_scores, comparator=comparators
        )
        self.assertEqual(errors.keys(), expected_errors.keys())
        for metric in errors:
            self.assertAlmostEqual(errors[metric], expected_errors[metric])

    def test_no_common_metrics(self):
        """Test the case where there are no common metrics between dictionaries."""
        real = {"roc_auc": 0.9}
        estimated = {"f1_score": 0.8}
        errors = score_error(real, estimated)
        self.assertEqual(errors, {})

    def test_invalid_comparator_type(self):
        """Test if a ValueError is raised for an invalid comparator type."""
        with self.assertRaises(ValueError):
            score_error(
                self.real_scores, self.estimated_scores, comparator="not_a_callable"
            )

    def test_empty_dictionaries(self):
        """Test the behavior with empty input dictionaries."""
        self.assertEqual(score_error({}, {}), {})
        self.assertEqual(score_error(self.real_scores, {}), {})
        self.assertEqual(score_error({}, self.estimated_scores), {})

    def test_verbose_output(self):
        """Test if the verbose mode prints the expected output."""
        with io.StringIO() as buf, redirect_stdout(buf):
            score_error(self.real_scores, self.estimated_scores, verbose=True)
            output = buf.getvalue()

        # Check if information for each common metric was printed
        self.assertIn("[accuracy] Real: 0.95, Estimated: 0.91", output)
        self.assertIn("[precision] Real: 0.9, Estimated: 0.92", output)
        # Check that non-common metrics (recall, f1_score) were not printed
        self.assertNotIn("recall", output)
        self.assertNotIn("f1_score", output)


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
