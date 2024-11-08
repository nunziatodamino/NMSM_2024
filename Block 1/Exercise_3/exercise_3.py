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

# --- Exercise 1

def integrandFunction (x):
    return np.sin(x)
  
left_bound = 0
right_bound = np.pi / 2
iterations = 1000
generated_points=1000

crude_mc_distribution=np.zeros(generated_points)

for i in range (generated_points):
    crude_mc_distribution[i]=mc.crude_monte_carlo(integrandFunction, left_bound, right_bound, iterations)

plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(6, 6))
plt.hist(crude_mc_distribution, bins=30, density=True, alpha=0.4, color='r')
plt.xlabel("Estimated Integral Value")
plt.ylabel("Frequency Density")
#plt.title(f"Monte Carlo Integration distribution using crude Monte Carlo method in {iterations} iterations")
image_name = "crude_mc.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.show()

# --- Importance sampling

# Choice of the sampling function
# Choosing the parameter a

a=1/np.pi
b = 24/np.pi**3 -(12 *a)/np.pi**2

def samplingFunction (x):
    return a + (b * x**2)

def original_sampling_function (x):
    return 1/(np.pi/2)

x = np.arange(0, np.pi/2, 0.1)

function_squared = integrandFunction(x)/original_sampling_function(x)
original_sampling=original_sampling_function(x)
new_sampling=samplingFunction(x)

fig, ax = plt.subplots(figsize=(6, 6))
plt.plot (x,function_squared, label=r"product $f^2\rho$")
plt.plot(x, new_sampling, label= "new sampling dist")
ax.axhline(y=2/np.pi, color='red', linestyle='-', linewidth=2, label="original sampling dist")
plt.legend(fontsize=10)
image_name = "choice_sampling_function.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.show()

mean = np.pi/2
sigma = 1
c = 1.4

def truncated_normal (x ,left_t, right_t):
    return stats.norm.pdf(x, mean, sigma) / (stats.norm.cdf(right_t)-stats.norm.cdf(left_t))

plt.plot (x, new_sampling , label = "target density")
plt.plot (x, c * truncated_normal (x, 0, np.pi/2)  ,label = "candidate density")

plt.legend()
plt.show()

loc=mean
scale=sigma
a_trunc=0
b_trunc=np.pi/2
a1, b1 = (a_trunc - loc) / scale, (b_trunc - loc) / scale


def importance_sampling_mc_vectorized(numberSampledPoints):
    total_sum = 0
    batch_size = 10 * numberSampledPoints 
    randomVariables = stats.truncnorm.rvs(a1, b1, loc=mean, scale=sigma, size=batch_size)
    acceptance_thresholds = samplingFunction(randomVariables) / (c * truncated_normal(randomVariables, a_trunc, b_trunc))       
    uniformRandomNums = np.random.random(batch_size)
    accepted_indices = uniformRandomNums <= acceptance_thresholds # boolean indexing of uniform_random_numbers
    accepted_samples = randomVariables[accepted_indices]
    accepted_samples = accepted_samples[accepted_samples != 0] # remove extra zeros
    if len(accepted_samples) > numberSampledPoints:
        accepted_samples = accepted_samples[:numberSampledPoints] 
    total_sum = np.sum(integrandFunction(accepted_samples) / samplingFunction(accepted_samples))                
    return total_sum / numberSampledPoints

imp_mc_distribution=np.zeros(generated_points)

for i in range (generated_points):
    imp_mc_distribution[i]=importance_sampling_mc_vectorized(iterations)

plt.hist(crude_mc_distribution, bins=30, density=True, alpha=0.4, color='r', label= "Crude Monte Carlo")
plt.hist(imp_mc_distribution, bins=30, density=True, alpha=0.4, color='b', label="Importance sampling")
plt.xlabel("Estimated Integral Value")
plt.ylabel("Frequency Density")
#plt.title(f"Monte Carlo Integration distribution comparison at {iterations} iterations")
plt.legend(fontsize=10)
plt.show()    
