import os
import numpy as np
import matplotlib.pyplot as plt
import multiple_histogram_method as mhm

########################################
file_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block_2/Exercise_12"
########################################

beta_list = np.load(os.path.join(file_path, "beta_list.npy"))
energy_results = np.load(os.path.join(file_path, "mmc_energy_series_10.npy")) 

N = 10

STEP = mhm.STEP

part = mhm.partition_function_recursion(beta_list, energy_results)
gen_part = mhm.partition_function_evaluation(beta_list, STEP, energy_results, part)
mean_energy_sq = mhm.mean_observable_evaluation(beta_list, STEP, energy_results**2, energy_results, part, gen_part)
mean_energy = mhm.mean_observable_evaluation(beta_list, STEP, energy_results, energy_results, part, gen_part)
energy_variance_dof = (mean_energy_sq - mean_energy**2) / N

beta_range = np.arange(beta_list[0], beta_list[-1] + STEP, STEP)
plt.plot(beta_range, energy_variance_dof)
plt.close()

max_index = np.argmax(energy_variance_dof)
beta_max = beta_range[max_index]
print(beta_max)
file_name = f"beta_{N}.npy"
np.save(file_name, beta_max)