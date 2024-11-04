import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate example data with 4 classes
np.random.seed(42)
n_samples = 400
n_features = 2
n_classes = 4

X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, n_classes, n_samples)

# Add some structure to the data
for i in range(n_classes):
    X[y == i] += [i * 2, i * 2]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the KNN model
k = 1  # number of neighbors
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

# Make predictions on the test set
y_pred = knn.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Visualize the results
plt.figure(figsize=(10, 8))

# Plot training data
colors = ['red', 'blue', 'green', 'purple']
for i in range(n_classes):
    plt.scatter(X_train[y_train == i][:, 0], X_train[y_train == i][:, 1], c=colors[i], label=f'Class {i} (Train)')

# Plot test data
for i in range(n_classes):
    plt.scatter(X_test[y_test == i][:, 0], X_test[y_test == i][:, 1], marker='s', s=80, facecolors='none', edgecolors=colors[i], label=f'Class {i} (Test)')

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title(f'KNN Classification with 4 Classes (k={k})')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
