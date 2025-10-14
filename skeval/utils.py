# Authors: The scikit-autoeval developers
# SPDX-License-Identifier: BSD-3-Clause
from sklearn.utils.validation import check_is_fitted as sklearn_check_is_fitted

def check_is_fitted(model):
    """Check if the model has been fitted.

    This method uses `sklearn.utils.validation.check_is_fitted` to verify
    that the underlying model has been fitted. It raises a `RuntimeError`
    if the model is not fitted.

    Raises
    ------
    RuntimeError
        If the model has not been fitted yet.
    """
    return sklearn_check_is_fitted(model)