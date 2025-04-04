import numpy as np
import matplotlib.pyplot as plt

numbers = range(10, 40, 5)

x_values = []
y_values = []

for num in numbers:
    filename = f'beta_{num}.npy'
    beta_value = np.load(filename)
    x_values.append(num**(-0.5))
    y_values.append(beta_value)

x_values = np.array(x_values)
y_values = np.array(y_values)

coefficients = np.polyfit(x_values, y_values, 1)
m, c = coefficients

best_fit_y = m * x_values + c

plt.figure(figsize=(8, 6))
plt.plot(x_values, y_values, 'o', label='Data Points', markersize=8)
plt.plot(x_values, best_fit_y, '-', label=f'Best Fit Line: y = {m:.3f}x + {c:.3f}', color='r')
plt.ylabel('Inverse temperature')
plt.xlabel('Number of monomers^(-1/2)')
plt.legend()
plt.grid(True)
plt.show()

