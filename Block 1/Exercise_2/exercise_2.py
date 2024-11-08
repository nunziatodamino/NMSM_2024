import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv
import scipy.stats as stats
import distribution_sampling as sampling

########################################
report_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Report"
exercise_folder = "FIG/exercise_2_images"
########################################

A = np.sqrt(2)/(np.e * kv(1/4,1))

def non_invertible_pdf (x):
    return A * np.exp ( -8*(x**2/2+x**4/4) )

mean = 0
sigma = np.sqrt(1/4)
c = 1.6

sample_space = np.arange(-3 ,+3, 0.01)
target_density = non_invertible_pdf(sample_space)

plt.rcParams.update({'font.size': 18})

plt.figure(figsize=(10, 6))
plt.plot (sample_space, target_density, label = "Target distribution")
plt.plot(sample_space, c *stats.norm.pdf(sample_space, mean, sigma), label = "Candidate density times constant c")
plt.xlabel("Random variable")
plt.ylabel("Probability density")
plt.legend(fontsize = 10)
image_name = "rejection_sampling_inequality_check.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.show()

NUMBER_SAMPLED_VARIABLES = 10000

random_variables = sampling.rejection_from_normal(non_invertible_pdf, c, mean, sigma, NUMBER_SAMPLED_VARIABLES)

plt.rcParams.update({'font.size': 18})

plt.figure(figsize=(10, 6)) # essential to avoid overcropping in the save process
plt.plot (sample_space, target_density, label = "Target distribution")
plt.hist(random_variables, bins=30, density=True, alpha=0.4, color='r', label= "Sampled variables")
plt.xlabel("Random variable")
plt.ylabel("Probability density")
plt.legend(fontsize = 10)
image_name = "rejection_sampling_verify.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.show()