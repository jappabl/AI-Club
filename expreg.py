import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

def exponential_func(x, a, b):
    return a * np.exp(b * x)

# Generate or load your data
x_data = np.array([0, 1, 2, 3, 4, 5])
y_data = np.array([2.0, 8, 16, 56, 256, 567])

# Perform the exponential regression
popt, _ = curve_fit(exponential_func, x_data, y_data)

# Extract the optimal parameters
a_opt, b_opt = popt

# Print the fitted equation
print(f"Fitted exponential equation: y = {a_opt:.4f} * e^({b_opt:.4f}x)")

# Generate points for plotting the fitted curve
x_fit = np.linspace(min(x_data), max(x_data), 100)
y_fit = exponential_func(x_fit, a_opt, b_opt)

# Plot the results
plt.scatter(x_data, y_data, label='Data')
plt.plot(x_fit, y_fit, 'r-', label='Fitted Curve')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Exponential Regression')
plt.show()
