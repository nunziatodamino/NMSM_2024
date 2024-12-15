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
energy_results = np.load(os.path.join(file_path, "energy_series.npy")) # N = 20

part = mhm.partition_function_recursion_vectorized(beta_list, energy_results)
print(part)

part = mhm.partition_function_recursion(beta_list, energy_results)
print(part)
