from __future__ import annotations

class PassThroughModel:
    """Dummy model used for tests and smoke runs.
    fit returns self; predict returns zeros of the right length.
    """

    def __init__(self, **kwargs):
        self.is_fitted = False

    def fit(self, X, y=None):
        self.is_fitted = True
        return self

    def predict(self, X):
        n = getattr(X, "shape", [len(X)])[0]
        return [0] * n

    def predict_proba(self, X):
        n = getattr(X, "shape", [len(X)])[0]
        # two-class uniform probs as a harmless default
        return [[0.5, 0.5] for _ in range(n)]
