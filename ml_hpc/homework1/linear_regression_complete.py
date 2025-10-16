import numpy as np


class LinearRegression:
    def fit(self, X, y, fit_intercept=True):
        """Fit Linear Regression model using closed-form solution."""
        self.fit_intercept = fit_intercept

        # Add intercept term
        if fit_intercept:
            ones = np.ones((X.shape[0], 1))
            X = np.concatenate((ones, X), axis=1)

        self.X = np.array(X)
        self.y = np.array(y)
        self.N, self.D = self.X.shape

        # === Estimate β parameters ===
        XtX = np.dot(self.X.T, self.X)
        XtX_inv = np.linalg.inv(XtX)
        Xty = np.dot(self.X.T, self.y)
        self.beta_hats = np.dot(XtX_inv, Xty)

        # === Predictions ===
        self.y_hat = np.dot(self.X, self.beta_hats)

        # === Training loss ===
        residuals = self.y - self.y_hat
        self.L = (1 / 2) * np.sum(residuals ** 2)

        return self

    def predict(self, X_test, predict_intercept=None):
        """Predict using the learned β parameters."""
        X_test = np.array(X_test)

        if predict_intercept is None:
            predict_intercept = self.fit_intercept
        if predict_intercept:
            ones = np.ones((X_test.shape[0], 1))
            X_test = np.concatenate((ones, X_test), axis=1)

        return np.dot(X_test, self.beta_hats)

    def get_params(self):
        """Return parameters in structured way."""
        if self.fit_intercept:
            return {
                'intercept': self.beta_hats[0],
                'coefficients': self.beta_hats[1:]
            }
        else:
            return {
                'intercept': 0.0,
                'coefficients': self.beta_hats
            }

