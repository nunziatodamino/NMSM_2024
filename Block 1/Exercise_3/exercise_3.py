import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import numba

# paths for saving images in report
########################################
report_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Report"
exercise_folder = "FIG/exercise_3_images"
########################################

@numba.njit
def integrand_function (x):
    return np.sin(x)

# ----- Crude Monte Carlo evaluation

@numba.njit
def crudeMonteCarlo (iterations, random_variable):
    sum = 0
    for _ in range(iterations):
        sum += integrand_function(random_variable)
    return (np.pi/2 -0 ) * sum/iterations    

ITERATIONS = 1000
GENERATED_POINTS=1000
crude_mc_distribution=np.zeros(GENERATED_POINTS)

for i in range (GENERATED_POINTS):
    crude_mc_distribution[i]=crudeMonteCarlo(ITERATIONS, np.random.uniform(0, np.pi/2))

plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(5, 3))
plt.hist(crude_mc_distribution, bins=30, density=True, alpha=0.4, color='r')
plt.xlabel("Estimated Integral Value")
plt.ylabel("Frequency Density")
#plt.title(f"Monte Carlo Integration distribution using crude Monte Carlo method in {ITERATIONS} ITERATIONS")
image_name = "crude_mc.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.show()

# --- Importance sampling

# Choosing the parameter a
a=1/np.pi
b = 24/np.pi**3 -(12 *a)/np.pi**2

@numba.njit
def proposed_sampling_function (x):
    return a + (b * x**2)

def original_sampling_function (x):
    return 1/(np.pi/2)

x = np.arange(0, np.pi/2, 0.1)

function_squared = integrand_function(x)/original_sampling_function(x)
original_sampling = original_sampling_function(x)
proposed_sampling = proposed_sampling_function(x)

fig, ax = plt.subplots(figsize=(6, 6))
plt.plot (x,function_squared, label=r"product $f^2\rho$")
plt.plot(x, proposed_sampling, label= "proposed sampling distribution")
ax.axhline(y=2/np.pi, color='red', linestyle='-', linewidth=2, label="original sampling distribution")
plt.legend(fontsize = 10)
image_name = "proposed_samplin_choice.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.show()

mean = np.pi/2
sigma = 1
c = 2

def truncated_normal (x, left_t, right_t):
    return stats.norm.pdf(x, mean, sigma) / (stats.norm.cdf(right_t) - stats.norm.cdf(left_t))

plt.plot (x, proposed_sampling , label = "target density")
plt.plot (x, c * truncated_normal (x, 0, np.pi/2)  ,label = "candidate density")

plt.legend()
plt.show()
plt.close()

#numberSampledPoints = 2000
loc=mean
scale=sigma
a_trunc=0
b_trunc=np.pi/2

a1, b1 = (a_trunc - loc) / scale, (b_trunc - loc) / scale

def importance_sampling_mc (numberSampledPoints):
    total_sum = 0
    counter = 0
    while counter < numberSampledPoints:
        randomVariable = stats.truncnorm.rvs(a1,b1, loc=mean, scale=sigma)
        uniformRandomNum = np.random.random()
        acceptance_threshold = proposed_sampling_function(randomVariable) / (c * truncated_normal(randomVariable, a_trunc,b_trunc) )
        if uniformRandomNum <= acceptance_threshold:
            total_sum+=integrand_function(randomVariable)/proposed_sampling_function(randomVariable)
            counter+=1
    return total_sum/counter       

imp_mc_distribution=np.zeros(GENERATED_POINTS)

for i in range (GENERATED_POINTS):
    imp_mc_distribution[i]=importance_sampling_mc(ITERATIONS)

plt.hist(crude_mc_distribution, bins=30, density=True, alpha=0.4, color='r', label= "Crude Monte Carlo")
plt.hist(imp_mc_distribution, bins=30, density=True, alpha=0.4, color='b', label="Importance sampling")
plt.xlabel("Estimated Integral Value")
plt.ylabel("Frequency Density")
#plt.title(f"Monte Carlo Integration distribution comparison at {ITERATIONS} ITERATIONS")
plt.legend(fontsize = 10)
image_name = "distribution_comparison.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.show()
