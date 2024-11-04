import numpy as np
import matplotlib.pyplot as plt
import pandas as pd  # Add this import
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import PartialDependenceDisplay

# Generate a random dataset
np.random.seed(42)
n_samples = 1000
n_features = 2
n_classes = 3

X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, n_classes, n_samples)

# Add some structure to the data
for i in range(n_classes):
    X[y == i] += [i * 2, i * 2]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the random forest model
n_estimators = 100
max_depth = 5
rf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
rf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualize the results
plt.figure(figsize=(15, 10))

# Plot the decision boundary
plt.subplot(121)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = rf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

colors = ['red', 'blue', 'green']
for i in range(n_classes):
    plt.scatter(X_train[y_train == i][:, 0], X_train[y_train == i][:, 1], c=colors[i], label=f'Class {i} (Train)')
    plt.scatter(X_test[y_test == i][:, 0], X_test[y_test == i][:, 1], marker='s', s=50, c=colors[i], alpha=0.6, label=f'Class {i} (Test)')

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Random Forest Classification')
plt.legend()

# Plot feature importance
plt.subplot(122)
importances = rf.feature_importances_
feature_names = [f'Feature {i+1}' for i in range(n_features)]
forest_importances = pd.Series(importances, index=feature_names)
forest_importances.plot.bar()
plt.title('Feature Importances')
plt.xlabel('Features')
plt.ylabel('Importance')

plt.tight_layout()
plt.show()

# Plot partial dependence plots
fig, ax = plt.subplots(figsize=(10, 6))
PartialDependenceDisplay.from_estimator(rf, X_train, [0, 1], ax=ax)
plt.tight_layout()
plt.show()
