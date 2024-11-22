import os
import numpy as np
import matplotlib.pyplot as plt
import importance_sampling as mc

########################################
report_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Report"
exercise_folder = "FIG/exercise_3_images"
########################################

def integrand_function (x):
    return np.sin(x)
  
LEFT_BOUND = 0
RIGHT_BOUND = np.pi / 2
ITERATIONS = 1000
GENERATED_POINTS = 1000

crude_mc_distribution = np.zeros(GENERATED_POINTS)
error_crude_mc = np.zeros(GENERATED_POINTS)

for i in range (GENERATED_POINTS):
    crude_mc_distribution[i], error_crude_mc[i] = mc.crude_monte_carlo_uniform(integrand_function, LEFT_BOUND, RIGHT_BOUND, ITERATIONS)

estimate = np.sum(crude_mc_distribution) / GENERATED_POINTS
error = np.sum(error_crude_mc) / GENERATED_POINTS

print(f"{estimate} $\pm$ {error} for the crude Monte Carlo")

plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(6, 6))
plt.hist(crude_mc_distribution, bins=30, density=True, alpha=0.4, color='r')
plt.xlabel("Estimated Integral Value")
plt.ylabel("Frequency Density")
#plt.title(f"Monte Carlo Integration distribution using crude Monte Carlo method in {ITERATIONS} ITERATIONS")
image_name = "crude_mc.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.show()

# --- Importance sampling
# Choice of the sampling function
# Choosing the parameter a

a=1 / np.pi
b = 24 / np.pi**3 - (12 * a) / np.pi**2

def sampling_function (x):
    return a + (b * x**2)

def original_sampling_function (x):
    return 1 / (np.pi / 2)

x = np.arange(0, np.pi / 2, 0.1)

function_squared = integrand_function(x) / original_sampling_function(x)
original_sampling = original_sampling_function(x)
new_sampling = sampling_function(x)

fig, ax = plt.subplots(figsize=(6, 6))
plt.plot (x,function_squared, label=r"product $f^2\rho$")
plt.plot(x, new_sampling, label= "new sampling dist")
ax.axhline(y=2/np.pi, color='red', linestyle='-', linewidth=2, label="original sampling dist")
plt.legend(fontsize=10)
image_name = "choice_sampling_function.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.show()

c=1.4
plt.plot (x, new_sampling , label = "target density")
plt.plot (x, c * mc.truncated_normal (x , np.pi/2, 1, 0, np.pi/2)  ,label = "candidate density")
plt.legend()
plt.show()

imp_mc_distribution = np.zeros(GENERATED_POINTS)
errors_imp_mc = np.zeros(GENERATED_POINTS)

for i in range(GENERATED_POINTS):
    imp_mc_distribution[i], errors_imp_mc[i] = mc.importance_sampling_mc_vectorized(integrand_function, sampling_function, ITERATIONS)

estimate = np.sum(imp_mc_distribution) / GENERATED_POINTS
error = np.sum(errors_imp_mc) / GENERATED_POINTS

print(f"{estimate} $\pm$ {error} for the importance sampling")    
    
fig, ax = plt.subplots(figsize=(6, 6))
plt.hist(crude_mc_distribution, bins=30, density=True, alpha=0.4, color='r', label= "Crude Monte Carlo")
plt.hist(imp_mc_distribution, bins=30, density=True, alpha=0.4, color='b', label="Importance sampling")
plt.xlabel("Estimated Integral Value")
plt.ylabel("Frequency Density")
#plt.title(f"Monte Carlo Integration distribution comparison at {ITERATIONS} ITERATIONS")
plt.legend(fontsize=10)
image_name = "distribution_comparison.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.show()