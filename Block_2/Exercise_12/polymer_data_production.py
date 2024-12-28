import os
import numpy as np
import polymer as polymer

########################################
file_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 2/Exercise_12"
########################################

time_start = 0
beta_list = np.load(os.path.join(file_path, "beta_list.npy"))

# Parameters
n_monomers = polymer.n_monomers
monomer_radius = polymer.monomer_radius
energy_threshold = polymer.energy_threshold
c = polymer.c

#initial configuration (vertical stick)
initial_polymer = np.zeros((n_monomers, 2))
bond = 1.15
for i in range (1, n_monomers):
    initial_polymer[i][1] = i * bond * c

b_block = 2500 
time_max = 40 * b_block
time_series = range(b_block, time_max + 1, b_block)

energy_evolution_results = np.zeros((len(beta_list), time_max ))
moves_beta = np.zeros((len(beta_list), time_max , polymer.n_monomers, 2))

initial_conf = np.zeros((len(beta_list), 1 , polymer.n_monomers, 2))
for beta in range(len(beta_list)):
    initial_conf[beta, 0] = initial_polymer

for i, time in enumerate(time_series):
    print(f"time : {time}...")
    for k, beta in enumerate(beta_list):
        start_idx = i * b_block
        end_idx = (i + 1) * b_block
        _, _,\
        energy_evolution_results[k][start_idx : end_idx], _, \
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

# Saving data
np.save(os.path.join(file_path, f"mmc_energy_series_{n_monomers}.npy"), energy_evolution_results)
