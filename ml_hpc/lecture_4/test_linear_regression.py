from linear_regression_incomplete import LinearRegression
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


diabetes = datasets.load_diabetes()
X = diabetes['data']
y = diabetes['target']

model = LinearRegression()  # instantiate model
model.fit(X, y)  # fit model

# Actual vs Predicted
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=model.y, y=model.y_hat, ax=ax, alpha=0.7)
# Perfect prediction line
min_val = min(min(model.y), min(model.y_hat))
max_val = max(max(model.y), max(model.y_hat))
ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
# Details
ax.set_xlabel(r'$y$ (Actual)', size=16)
ax.set_ylabel(r'$\hat{y}$ (Predicted)', rotation=0, size=16, labelpad=15)
ax.set_title(r'Actual vs Predicted Values', size=20, pad=10)
ax.legend()
sns.despine()
plt.tight_layout()
plt.savefig('actual_vs_predicted_diabetes.png', dpi=300, bbox_inches='tight')
plt.close()


# With fake data
np.random.seed(42)
n_samples = 1000
n_features = 3
X_multi = np.random.randn(n_samples, n_features) * 5
true_coeffs = [1.5, -2.0, 0.8]
true_intercept = 2.0
y_multi = true_intercept + np.dot(X_multi, true_coeffs) + np.random.randn(n_samples) * 1.5

model_multi = LinearRegression()
model_multi.fit(X_multi, y_multi)


# Actual vs Predicted
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(x=model_multi.y, y=model_multi.y_hat, ax=ax, alpha=0.7)
# Perfect prediction line
min_val = min(min(model_multi.y), min(model_multi.y_hat))
max_val = max(max(model_multi.y), max(model_multi.y_hat))
ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
# Details
ax.set_xlabel(r'$y$ (Actual)', size=16)
ax.set_ylabel(r'$\hat{y}$ (Predicted)', rotation=0, size=16, labelpad=15)
ax.set_title(r'Actual vs Predicted Values', size=20, pad=10)
ax.legend()
sns.despine()
plt.tight_layout()
plt.savefig('actual_vs_predicted_numpy.png', dpi=300, bbox_inches='tight')
plt.close()
