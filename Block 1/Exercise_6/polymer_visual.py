import os
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf
import polymer as polymer

########################################
report_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Report"
file_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Exercise_6"
exercise_folder = "FIG/exercise_6_images"
########################################

n_monomers = polymer.n_monomers
monomer_radius = polymer.monomer_radius
energy_threshold = polymer.energy_threshold
c = polymer.c

beta_list = np.load(os.path.join(file_path, "beta_list.npy"))
energy_evolution_results = np.load(os.path.join(file_path, "mega_energy_series.npy"))
ee2_results = np.load(os.path.join(file_path, "mega_ee2_series.npy"))
end_height_results = np.load(os.path.join(file_path, "mega_end2height_series.npy"))
gyr_rad_results = np.load(os.path.join(file_path, "mega_gyr_radius_series.npy"))

#initial configuration (vertical stick)
monomers_initial_conf=np.zeros((n_monomers,2))
bond = 1.15
for i in range (1, n_monomers):
    monomers_initial_conf[i][1]=i*bond*c

# Initial configuration plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal', adjustable='datalim')

for i, point in enumerate(monomers_initial_conf):
    circle = plt.Circle((point[0], point[1]), 0.5, color='dodgerblue', alpha=1,linewidth=1.5)
    ax.add_patch(circle)
    if i > 0:
        prev_point = monomers_initial_conf[i - 1]
        ax.plot([prev_point[0], point[0]], [prev_point[1], point[1]], color='dodgerblue', linewidth=2)

ax.set_xlim(min(p[0] for p in monomers_initial_conf) - 1, max(p[0] for p in monomers_initial_conf) + 1)
ax.set_ylim(min(p[1] for p in monomers_initial_conf) - 1, max(p[1] for p in monomers_initial_conf) + 1)
ax.axhline(y=-monomer_radius, color='red', linestyle='-', linewidth=2, label="Wall")
ax.axhline(y=energy_threshold, color='purple', linestyle=':', linewidth=1.5, label="Energy Threshold")
ax.set_xlabel("X-axis", fontsize=12, labelpad=10, color='darkblue')
ax.set_ylabel("Y-axis", fontsize=12, labelpad=10, color='darkblue')
ax.set_title("Initial Polymer Configuration", fontsize=14, color='darkblue', pad=15)
#plt.show()
plt.close()

# A move series example
total_moves=15

configurations=np.zeros((total_moves, n_monomers, 2))
configurations[0]=monomers_initial_conf

for i in range(1,total_moves):
    configurations[i]=polymer.polymer_displacement(configurations[i-1])

conf=configurations[total_moves-1]

fig, ax = plt.subplots(figsize=(6, 6))
ax.set_aspect('equal', adjustable='datalim')

for i, point in enumerate(conf):
    circle = plt.Circle((point[0], point[1]), 0.5, color='dodgerblue', alpha=1,linewidth=1.5)
    ax.add_patch(circle)
    if i > 0:
        prev_point = conf[i - 1]
        ax.plot([prev_point[0], point[0]], [prev_point[1], point[1]], color='dodgerblue', linewidth=2)

ax.set_xlim(min(p[0] for p in conf) - 1, max(p[0] for p in conf) + 1)
ax.set_ylim(min(p[1] for p in conf) - 1, max(p[1] for p in conf) + 1)
ax.axhline(y=-monomer_radius, color='red', linestyle='-', linewidth=2, label="Wall")
ax.axhline(y=energy_threshold, color='purple', linestyle=':', linewidth=1.5, label="Energy Threshold")
ax.set_xlabel("X-axis", fontsize=12, labelpad=10, color='darkblue')
ax.set_ylabel("Y-axis", fontsize=12, labelpad=10, color='darkblue')
ax.set_title(f"Move Configuration after {total_moves} Moves", fontsize=14, color='darkblue', pad=15)
#plt.show()
plt.close()

#Energy histogram
plt.figure(figsize=(9, 6))
bins = np.arange(-n_monomers, 0, 0.5)
colors = plt.cm.tab10.colors 
bar_width = 0.1  
for i, beta in enumerate(beta_list):
    hist, bin_edges = np.histogram(energy_evolution_results[i], bins=bins)
    hist = hist / hist.sum()     
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    offset = (i - len(beta_list) / 2) * bar_width  
    for j, center in enumerate(bin_centers):
        plt.vlines(center + offset, 0, hist[j], color=colors[i % len(colors)], 
                   label=f'beta={beta:.2f}' if j == 0 else None, alpha=0.8)
plt.xlabel("Energy")
plt.ylabel("Normalized Frequency")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))  
plt.tight_layout() 
image_name = "energy_histogram.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.close()
