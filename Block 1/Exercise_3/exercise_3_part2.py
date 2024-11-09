import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import numba
import importance_sampling as mc

# paths for saving images in report
########################################
report_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Report"
exercise_folder = "FIG/exercise_3_images"
########################################

def integration_function(x):
    return np.exp(-(x - 3)**2 /2) + np.exp(-(x - 6)**2 /2)

ITERATIONS = 1000
GENERATED_POINTS=1000
MEAN = 0
SIGMA = 1

crude_mc_distribution = np.zeros(GENERATED_POINTS)
errors_crude_mc = np.zeros(GENERATED_POINTS)

for i in range (GENERATED_POINTS):
    crude_mc_distribution[i], errors_crude_mc[i] = mc.crude_monte_carlo_normal(integration_function, MEAN, SIGMA, ITERATIONS)

plt.rcParams.update({'font.size': 18})
fig, ax = plt.subplots(figsize=(6, 6))
plt.hist(crude_mc_distribution, bins=30, density=True, alpha=0.4, color='r')
plt.xlabel("Estimated Integral Value")
plt.ylabel("Frequency Density")
image_name = "distribution_crude_mc_ex2.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.show()

estimate = np.sum(crude_mc_distribution) / GENERATED_POINTS
error = np.sum(errors_crude_mc) / GENERATED_POINTS

print(f"{estimate} $\pm$ {error} for the crude Monte Carlo")

x = np.arange(-10, 10, 0.1)
y = integration_function(x) 

fig, ax = plt.subplots(figsize=(10, 10))
plt.plot(x, y, label=r"$f(x)$")
plt.plot(x, stats.norm.pdf(x, 0, 1), label="Normal Distribution (mean=0, sigma=1)")
plt.plot(x, integration_function(x) * stats.norm.pdf(x, 0, 1), label="Integrand Function")
plt.hlines(1/7, -8, -1, color='purple', linestyle='-', linewidth=2, label=r"$U(-8,-1)$")
plt.vlines([-8, -1], ymin=0, ymax=1/7, color='purple', linestyle='dotted')
plt.xlabel("x")
plt.ylabel("y")
plt.legend(fontsize = 10)
image_name = "distribution_consideration.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.show()

def importance_monte_carlo(iterations):
    values = np.random.uniform(-1, 4.5, size = iterations)
    func_values = (integration_function(values) * stats.norm.pdf(values, MEAN, SIGMA)) / (1/(5.5))
    estimate = np.sum(func_values) / iterations
    variance = np.sum(func_values**2) / iterations - estimate**2 
    return estimate, np.sqrt(variance / iterations)

importance_mc_distribution=np.zeros(GENERATED_POINTS)
errors_importance_mc=np.zeros(GENERATED_POINTS)

for i in range (GENERATED_POINTS):
    importance_mc_distribution[i], errors_importance_mc = importance_monte_carlo(ITERATIONS)

fig, ax = plt.subplots(figsize=(10, 10))
plt.hist(crude_mc_distribution, bins=30, density=True, alpha=0.4, color='r')
plt.hist(importance_mc_distribution, bins=30, density=True, alpha=0.4, color='b')
plt.xlabel("Estimated Integral Value")
plt.ylabel("Frequency Density")
#plt.title(f"Monte Carlo Integration distribution using crude Monte Carlo method in {iterations} iterations")
image_name = "distribution_comparison_part2.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.show()

estimate = np.sum(importance_mc_distribution) / GENERATED_POINTS
error = np.sum(errors_importance_mc) / GENERATED_POINTS

print(f"{estimate} $\pm$ {error} for the importance sampling")

value1 = integration_function(-8) * stats.norm.pdf(-8, MEAN, SIGMA) 
value2 = integration_function(-1) * stats.norm.pdf(-1, MEAN, SIGMA) 

print(f"{value1} to {value2}")