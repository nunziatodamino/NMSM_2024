import os
import numpy as np
import matplotlib.pyplot as plt
import multiple_histogram_method as mhm

########################################
#report_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Report"
file_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 2/Exercise_12"
#exercise_folder = "FIG/exercise_6_images"
########################################

beta_list = np.load(os.path.join(file_path, "beta_list.npy"))
energy_results = np.load(os.path.join(file_path, "mmc_energy_series_20.npy")) # N = 20

STEP = mhm.STEP

part = mhm.partition_function_recursion(beta_list, energy_results)
print(part)
gen_part = mhm.partition_function_evaluation(beta_list, STEP, energy_results, part)
print(gen_part)
mean_energy = mhm.mean_observable_evaluation(beta_list, STEP, energy_results, energy_results, part, gen_part)

beta_range = np.arange(beta_list[0], beta_list[-1] + STEP, STEP)
plt.plot(beta_range, mean_energy)
plt.show()