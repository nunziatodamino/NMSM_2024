import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit
import polymer as polymer

########################################
report_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Report"
file_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Exercise_6"
exercise_folder = "FIG/exercise_6_images"
########################################

time_start = np.load(os.path.join(file_path, "time_max.npy"))
beta_list = np.load(os.path.join(file_path, "beta_list.npy"))

moves_initial = np.load(os.path.join(file_path, f"final_conf.npy"))

b_block = 25000 # maximum tau in k series
time_max = 40 * b_block
time_series = range(b_block, time_max + 1, b_block)

ee2_results = np.zeros((len(beta_list), time_max ))
end_height_results = np.zeros((len(beta_list), time_max ))
energy_evolution_results = np.zeros((len(beta_list), time_max ))
gyr_rad_results = np.zeros((len(beta_list), time_max ))
moves_beta = np.zeros((len(beta_list), time_max , polymer.n_monomers, 2))

initial_conf = np.zeros((len(beta_list), 1 , polymer.n_monomers, 2))
initial_conf = moves_initial[: , 0]

for i, time in enumerate(time_series):
    print(f"time : {time}...")
    for k, beta in enumerate(beta_list):
        start_idx = i * b_block
        end_idx = (i + 1) * b_block
        ee2_results[k][start_idx : end_idx], \
        end_height_results[k][start_idx : end_idx], \
        energy_evolution_results[k][start_idx : end_idx],\
        gyr_rad_results[k][start_idx : end_idx], \
        moves_beta[k][start_idx : end_idx] = polymer.thermalization(initial_conf[k], b_block , beta)        
        #print(f"siamo alla fine del {k+1}° beta")
        initial_conf[k] = moves_beta[k][end_idx - 1]
    k_selected = np.random.randint(0, len(beta_list) - 2)
    #print(f"Attempt swap at {k_selected}...")
    if polymer.mmc_swap(moves_beta[k_selected][-1], moves_beta[k_selected + 1][-1], beta_list, k_selected):
        tmp = moves_beta[k_selected + 1][-1]
        moves_beta[k_selected + 1][-1] = moves_beta[k_selected][-1]
        moves_beta[k_selected][-1] = tmp
        #print("Swap !")

# Observables plot
time = np.arange(0, time_max, 1)
selected_indices = np.linspace(0, len(beta_list) - 1, 10, dtype=int)
colors = plt.cm.tab10(np.linspace(0, 1, 4))
fig, axes = plt.subplots(5, 2, figsize=(30, 22))
axes = axes.flatten()
for ax, i in zip(axes, selected_indices):
    ax.plot(time, ee2_results[i], label='End-to-End', color=colors[0], linewidth=2)
    ax.plot(time, end_height_results[i], label='End Height', color=colors[1], linewidth=2)
    ax.plot(time, energy_evolution_results[i], label='Energy', color=colors[2], linewidth=2)
    ax.plot(time, gyr_rad_results[i], label='Gyration Radius', color=colors[3], linewidth=2)
    ax.set_title(f'Beta = {beta_list[i]:.2f}', fontsize=14)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Observables", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=20, frameon=True)
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.1, wspace=0.1, hspace=0.4)
image_name = "observables_mmc_polymer_summary.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.suptitle("Evolution of observables for selected beta values", fontsize=25)
plt.savefig(entire_path)
plt.close()

# Saving data
np.save(os.path.join(file_path, f"mmc_final_conf.npy"), moves_beta)
np.save(os.path.join(file_path, f"mmc_energy_series.npy"), energy_evolution_results)
np.save(os.path.join(file_path, f"mmc_ee2_series.npy"), ee2_results)
np.save(os.path.join(file_path, f"mmc_end2height_series.npy"),end_height_results)
np.save(os.path.join(file_path, f"mmc_gyr_radius_series.npy"), gyr_rad_results)
