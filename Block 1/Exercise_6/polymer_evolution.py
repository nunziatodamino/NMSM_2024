import os
import numpy as np
import matplotlib.pyplot as plt
import polymer as polymer

########################################
report_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Report"
file_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Exercise_6"
exercise_folder = "FIG/exercise_6_images"
########################################

# Parameters
n_monomers = polymer.n_monomers
monomer_radius = polymer.monomer_radius
energy_threshold = polymer.energy_threshold
c = polymer.c

#initial configuration (vertical stick)
monomers_initial_conf = np.zeros((n_monomers, 2))
bond = 1.15
for i in range (1, n_monomers):
    monomers_initial_conf[i][1] = i * bond * c

# Thermalization
time_max = 1000000

BETA_DOWN = 2
BETA_UP = 4.5
beta_list = np.linspace(0, BETA_DOWN, 10 * BETA_DOWN) 
beta_additional_1 = np.arange(BETA_DOWN + 0.5, BETA_UP, 0.5)
beta_list = np.sort(np.concatenate((beta_list, beta_additional_1) , axis=None)) 

ee2_results = np.zeros((len(beta_list), time_max))
end_height_results = np.zeros((len(beta_list), time_max))
energy_evolution_results = np.zeros((len(beta_list), time_max))
gyr_radius_results = np.zeros((len(beta_list), time_max))
moves_beta = np.zeros((len(beta_list), time_max, n_monomers, 2))

for k, beta in enumerate(beta_list):
    ee2_results[k], end_height_results[k], energy_evolution_results[k], gyr_radius_results[k], moves_beta[k] = polymer.thermalization(monomers_initial_conf, time_max, beta)
    print(f"siamo alla fine del {k+1}° beta")


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
    ax.plot(time, gyr_radius_results[i], label='Gyration Radius', color=colors[3], linewidth=2)
    ax.set_title(f'Beta = {beta_list[i]:.2f}', fontsize=14)
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Observables", fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=10)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=4, fontsize=20, frameon=True)
plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.1, wspace=0.1, hspace=0.4)
image_name = "observables_polymer_summary.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.suptitle("Evolution of observables for selected beta values", fontsize=25)
plt.savefig(entire_path)
plt.close()

# Energy histogram
for i, beta in enumerate(beta_list):
    bins = np.arange(-n_monomers, 0, 0.5)
    plt.figure(figsize = (9, 6))
    plt.hist(energy_evolution_results[i], bins=bins, density=True, alpha=0.4, label=f'beta={beta_list[i]}')
plt.xlabel("Energy")
plt.ylabel("Normalised frequency")
image_name = f"energy_histogram.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.close()

# Energy distribution overlap check
def dist_intersection_percentage(dist1, dist2):
    return np.sum(np.minimum(dist1, dist2)) / (np.sum(dist1) + np.sum(dist2))

bins = np.arange(-n_monomers, 0, 0.5)
for i in range(1,len(beta_list)):
    hist1, _ = np.histogram(energy_evolution_results[i], bins=bins, density=True)
    hist2, _ = np.histogram(energy_evolution_results[i-1], bins=bins, density=True)
    overlap = dist_intersection_percentage(hist1, hist2)
    print(f"Overlap between {beta_list[i-1]} and {beta_list[i]} of {overlap * 100} %")

# Saving data
np.save(os.path.join(file_path, f"mega_final_conf.npy"), moves_beta)
np.save(os.path.join(file_path, f"mega_time_max.npy"), time_max)
#np.save(os.path.join(file_path, f"beta_list.npy"), beta_list)
np.save(os.path.join(file_path, f"mega_energy_series.npy"), energy_evolution_results)
np.save(os.path.join(file_path, f"mega_ee2_series.npy"), ee2_results)
np.save(os.path.join(file_path, f"mega_end2height_series.npy"),end_height_results)
np.save(os.path.join(file_path, f"mega_gyr_radius_series.npy"), gyr_radius_results)
