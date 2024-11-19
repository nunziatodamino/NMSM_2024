import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import polymer_mmc as polymer

########################################
report_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Report"
file_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Exercise_6"
exercise_folder = "FIG/exercise_6_images"
########################################

time_max = np.load(os.path.join(file_path, "----"))
beta_list = np.load(os.path.join(file_path, "----"))
energy_evolution_results = np.load(os.path.join(file_path, "----"))
ee2_results = np.load(os.path.join(file_path, "----"))
end_height_results = np.load(os.path.join(file_path, "----"))

energy_autocorr=np.zeros((len(beta_list), time_max))
end2end_autocorr=np.zeros((len(beta_list), time_max))
end_height_autocorr=np.zeros((len(beta_list), time_max))

t_equilibrium = 2000
time = np.arange(0,time_max,1)
for i, beta in enumerate(beta_list):
    energy_autocorr[i] = acf(energy_evolution_results[i][t_equilibrium:],nlags=time_max)
    plt.plot(time,energy_autocorr[i], label=f"beta={beta}")

plt.title("Autocorrelation function for the energy at different betas")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.close() 

for i, beta in enumerate(beta_list):
    end2end_autocorr[i] = acf(ee2_results[i][t_equilibrium:],nlags=time_max)
    plt.plot(time,end2end_autocorr[i], label=f"beta={beta}")

plt.title("Autocorrelation function for the end-to-end distance at different betas")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.close() 

for i, beta in enumerate(beta_list):
    end_height_autocorr[i] = acf(end_height_results[i][t_equilibrium:],nlags=time_max)
    plt.plot(time, end_height_autocorr[i], label=f"beta={beta}")

plt.title("Autocorrelation function for the end height at different betas")
plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
plt.close() 
