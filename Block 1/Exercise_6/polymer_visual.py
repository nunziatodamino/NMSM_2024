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

n_monomers = polymer.n_monomers
monomer_radius = polymer.monomer_radius
energy_threshold = polymer.energy_threshold
c = polymer.c

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

plt.show()

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

plt.show()
