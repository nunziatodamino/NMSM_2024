import matplotlib.pyplot as plt
import ellipsoid_mc as mc
import numpy as np

# program parameters
X_SEMI_AXIS = 3.0
Y_SEMI_AXIS = 2.0
Z_SEMI_AXIS = 2.0
ITERATIONS = 10000

SAMPLES = 10000
distribution_first_ellipsoid=np.zeros(SAMPLES)
for i in range(SAMPLES):
    distribution_first_ellipsoid[i] = mc.monte_carlo_ellipsoid_octant(X_SEMI_AXIS, Y_SEMI_AXIS, Z_SEMI_AXIS, ITERATIONS)

plt.hist(distribution_first_ellipsoid, bins=30, density=True, alpha=0.4, color='r', label = f"a ={X_SEMI_AXIS}, b = {Y_SEMI_AXIS}, c = {Z_SEMI_AXIS}")
plt.title(f"Normalized distribution of the MC volume for k={ITERATIONS} steps")
plt.ylabel("Probability density")
plt.legend()
plt.xlabel("Volume")
plt.show()

# New ellipsoid parameters
X_SEMI_AXIS_NEW = 3.0
Y_SEMI_AXIS_NEW = 1.0
Z_SEMI_AXIS_NEW = 1.0

distribution_second_ellipsoid=np.zeros(SAMPLES)
for i in range(SAMPLES):
    distribution_second_ellipsoid[i] = mc.monte_carlo_ellipsoid_octant(X_SEMI_AXIS_NEW, Y_SEMI_AXIS_NEW, Z_SEMI_AXIS_NEW, ITERATIONS)

plt.hist(distribution_first_ellipsoid, bins=30, density=True, alpha=0.4, color='r', label = f"a ={X_SEMI_AXIS}, b = {Y_SEMI_AXIS}, c = {Z_SEMI_AXIS}")
plt.hist(distribution_second_ellipsoid, bins=30, density=True, alpha=0.4, color='b', label = f"a ={X_SEMI_AXIS_NEW}, b = {Y_SEMI_AXIS_NEW}, c = {Z_SEMI_AXIS_NEW}")
plt.title(f"Normalized distributions of the MC volume for k={ITERATIONS} steps comparison")
plt.ylabel("Probability density")
plt.xlabel("Volume")
plt.legend()
plt.show()

# Error plot as a function of the iterations
iteration_list = np.linspace(100, 100000, 100)
relative_errors = np.zeros(len(iteration_list))
relative_errors_new = np.zeros(len(iteration_list))
for i, iteration in enumerate(iteration_list):
    volume_estimate = mc.monte_carlo_ellipsoid_octant(X_SEMI_AXIS, Y_SEMI_AXIS, Z_SEMI_AXIS, iteration)
    relative_errors[i] = abs(volume_estimate - mc.analytic_ellipsoid_octant(X_SEMI_AXIS, Y_SEMI_AXIS, Z_SEMI_AXIS))/mc.analytic_ellipsoid_octant(X_SEMI_AXIS, Y_SEMI_AXIS, Z_SEMI_AXIS)
    volume_estimate_new = mc.monte_carlo_ellipsoid_octant(X_SEMI_AXIS_NEW, Y_SEMI_AXIS_NEW, Z_SEMI_AXIS_NEW, iteration)
    relative_errors_new[i] = abs(volume_estimate_new - mc.analytic_ellipsoid_octant(X_SEMI_AXIS_NEW, Y_SEMI_AXIS_NEW, Z_SEMI_AXIS_NEW))/mc.analytic_ellipsoid_octant(X_SEMI_AXIS_NEW, Y_SEMI_AXIS_NEW, Z_SEMI_AXIS_NEW)

plt.plot(iteration_list, relative_errors, label = f"a ={X_SEMI_AXIS}, b = {Y_SEMI_AXIS}, c = {Z_SEMI_AXIS}" )
plt.plot(iteration_list, relative_errors_new, label = f"a ={X_SEMI_AXIS_NEW}, b = {Y_SEMI_AXIS_NEW}, c = {Z_SEMI_AXIS_NEW}" )
plt.plot(iteration_list, 1/np.sqrt(iteration_list), label = "Theoretical curve")
plt.title("Comparison between the deviations from the analytical value as a function of the number of steps between the two ellipsoids")
plt.ylabel("Relative error")
plt.xlabel("Number of steps")
plt.legend()
plt.show()
