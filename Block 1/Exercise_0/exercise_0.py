import os
import matplotlib.pyplot as plt
import ellipsoid_mc as mc
import numpy as np

# paths for saving images in report
########################################
report_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Report"
exercise_folder = "FIG/exercise_0_images"
########################################

# program parameters
X_SEMI_AXIS = 3.0
Y_SEMI_AXIS = 2.0
Z_SEMI_AXIS = 2.0
ITERATIONS = 10000

SAMPLES = 10000
distribution_first_ellipsoid=np.zeros(SAMPLES)
error_distr_first=np.zeros(SAMPLES)
for i in range(SAMPLES):
    distribution_first_ellipsoid[i], error_distr_first[i] = mc.monte_carlo_ellipsoid_octant(X_SEMI_AXIS, Y_SEMI_AXIS, Z_SEMI_AXIS, ITERATIONS)

estimate = np.sum(distribution_first_ellipsoid) / SAMPLES
error = np.sum(error_distr_first) / SAMPLES

print(f"Volume : {estimate} $\pm$ {error}")

plt.rcParams.update({'font.size': 18}) # global font parameter for plots
plt.figure(figsize=(10, 10))
plt.hist(distribution_first_ellipsoid, bins=30, density=True, alpha=0.4, color='r', label = f"a ={X_SEMI_AXIS}, b = {Y_SEMI_AXIS}, c = {Z_SEMI_AXIS}")
#plt.title(f"Normalized distribution of the MC volume for k={ITERATIONS} steps")
plt.ylabel("Probability density")
plt.legend(fontsize = 10)
plt.xlabel("Volume")
image_name = "first_ellipsoid_distribution.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.close()

# New ellipsoid parameters
X_SEMI_AXIS_NEW = 3.0
Y_SEMI_AXIS_NEW = 1.0
Z_SEMI_AXIS_NEW = 1.0

distribution_second_ellipsoid=np.zeros(SAMPLES)
error_distr_second=np.zeros(SAMPLES)

for i in range(SAMPLES):
    distribution_second_ellipsoid[i], error_distr_second[i] = mc.monte_carlo_ellipsoid_octant(X_SEMI_AXIS_NEW, Y_SEMI_AXIS_NEW, Z_SEMI_AXIS_NEW, ITERATIONS)

estimate = np.sum(distribution_second_ellipsoid) / SAMPLES
error = np.sum(error_distr_second) / SAMPLES

print(f"Volume : {estimate} $\pm$ {error}")

plt.figure(figsize=(10, 10))
plt.hist(distribution_first_ellipsoid, bins=30, density=True, alpha=0.4, color='r', label = f"a ={X_SEMI_AXIS}, b = {Y_SEMI_AXIS}, c = {Z_SEMI_AXIS}")
plt.hist(distribution_second_ellipsoid, bins=30, density=True, alpha=0.4, color='b', label = f"a ={X_SEMI_AXIS_NEW}, b = {Y_SEMI_AXIS_NEW}, c = {Z_SEMI_AXIS_NEW}")
#plt.title(f"Normalized distributions of the MC volume for k={ITERATIONS} steps comparison")
plt.ylabel("Probability density")
plt.xlabel("Volume")
plt.legend(fontsize = 10)
image_name = "first_and_second_ellipsoid_distribution.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.close()

# Error plot as a function of the iterations
iteration_list = np.linspace(100, 100000, 100)
deviations = np.zeros(len(iteration_list))
deviations_new = np.zeros(len(iteration_list))
for i, iteration in enumerate(iteration_list):
    volume_estimate , _ = mc.monte_carlo_ellipsoid_octant(X_SEMI_AXIS, Y_SEMI_AXIS, Z_SEMI_AXIS, iteration)
    deviations[i] = abs(volume_estimate - mc.analytic_ellipsoid_octant(X_SEMI_AXIS, Y_SEMI_AXIS, Z_SEMI_AXIS))
    volume_estimate_new, _ = mc.monte_carlo_ellipsoid_octant(X_SEMI_AXIS_NEW, Y_SEMI_AXIS_NEW, Z_SEMI_AXIS_NEW, iteration)
    deviations_new[i] = abs(volume_estimate_new - mc.analytic_ellipsoid_octant(X_SEMI_AXIS_NEW, Y_SEMI_AXIS_NEW, Z_SEMI_AXIS_NEW))

plt.figure(figsize=(10, 10))
plt.plot(iteration_list, deviations, label = f"a ={X_SEMI_AXIS}, b = {Y_SEMI_AXIS}, c = {Z_SEMI_AXIS}" )
plt.plot(iteration_list, deviations_new, label = f"a ={X_SEMI_AXIS_NEW}, b = {Y_SEMI_AXIS_NEW}, c = {Z_SEMI_AXIS_NEW}")
plt.plot(iteration_list, 1 / np.sqrt(iteration_list), label = "Theoretical curve")
#plt.title("Comparison between the deviations from the analytical value as a function of the number of steps between the two ellipsoids")
plt.ylabel("Deviation from the analytical value")
plt.xlabel("Number of steps")
plt.legend(fontsize = 10)
image_name = "relative_error_trend.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.close()
