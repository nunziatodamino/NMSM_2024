import os
import numpy as np
import matplotlib.pyplot as plt
import ising_2d as ising

plt.rcParams.update({'font.size': 18}) # global font parameter for plots

# paths for saving images in report
########################################
report_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Report"
file_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Exercise_5"
exercise_folder = "FIG/exercise_5_images"
########################################

# Model parameters
LENGTH = 25
CRITICAL_TEMP = 2 / np.log(1 + np.sqrt(2))
BETA_CRITICAL = 1 / CRITICAL_TEMP
TEMP_MULT = (0.5, 0.95, 1.05, 2.5)
TEMP_LIST = np.array(TEMP_MULT) * CRITICAL_TEMP
BETA_LIST = 1 / TEMP_LIST

# Initial configuration

np.random.seed(69420)

SPIN_UP = 1
SPIN_DOWN = -1

initial_configuration = np.random.choice([SPIN_UP, SPIN_DOWN], size=(LENGTH, LENGTH))

np.random.seed(None) # in this way the initial configuration is set always in the same way, but the rest of the evolution no

plt.figure(figsize=(9, 6))
plt.imshow(initial_configuration, cmap="coolwarm", interpolation="nearest", origin="lower")
cbar = plt.colorbar(ticks=[SPIN_DOWN, SPIN_UP]) 
cbar.set_label("Spin")
plt.title("Initial Spin Configuration")
image_name = f"initial_spin_{LENGTH}.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.close()

neighbors_list = ising.neighbors_list_square_pbc_opt(LENGTH) # the neighbor list is configuration independent and can be evaluated at the start of the procedure.   

MC_TIMESTEPS = 100000
TIMESTEPS = LENGTH * LENGTH

#configurations = np.zeros((len(BETA_LIST), MC_TIMESTEPS, LENGTH, LENGTH))
energy_per_spin = np.zeros((len(BETA_LIST), MC_TIMESTEPS))
magnetisation_per_spin = np.zeros((len(BETA_LIST), MC_TIMESTEPS))
config = initial_configuration  

for temp, beta in enumerate(BETA_LIST):
    for mc_step in range(MC_TIMESTEPS):
        for _ in range(TIMESTEPS):
            config = ising.metropolis_spin_flip_dynamics_opt(config, neighbors_list, LENGTH, beta)
        energy_per_spin[temp, mc_step] = ising.system_energy_opt(config, neighbors_list, LENGTH) / (LENGTH * LENGTH)
        magnetisation_per_spin[temp, mc_step] = np.sum(config) / (LENGTH * LENGTH)

# # Plot the final configuration 
# cmap = mcolors.ListedColormap(["blue", "red"])
# bounds = [-1, 0, 1]  
# norm = mcolors.BoundaryNorm(bounds, cmap.N)
# plt.imshow(configurations[MC_TIMESTEPS-1], cmap=cmap, norm=norm, interpolation="nearest")
# cbar = plt.colorbar(ticks=[-1, 1])  
# cbar.set_label("Spin")
# cbar.ax.set_yticklabels(["Spin Down", "Spin Up"]) 
# plt.title("Spin Configuration")
# plt.show()

# Thermalization

time = np.arange(0, MC_TIMESTEPS, 1)
    
for j, beta in enumerate(BETA_LIST):
    plt.figure(figsize=(10, 10))
    plt.plot(time, energy_per_spin[j], label = f"energy per spin at {1/beta/CRITICAL_TEMP} T_c")
    plt.plot(time, magnetisation_per_spin[j], label = f"magnetisation per spin at {1/beta/CRITICAL_TEMP} T_c")
    plt.xlabel("Time in MC timestep units")
    plt.ylabel("Observables")
    plt.ylim(-3, 3) 
    plt.legend(fontsize = 10)
    image_name = f"thermalization_temp{1/beta/CRITICAL_TEMP:.2f}T_c_dimension{LENGTH}.png"
    entire_path = os.path.join(report_path, exercise_folder, image_name)
    plt.savefig(entire_path)
    plt.close()

np.save(os.path.join(file_path, f"critical_temp.npy"), CRITICAL_TEMP)
np.save(os.path.join(file_path, f"beta_list.npy"), BETA_LIST)
np.save(os.path.join(file_path, f"mc_timesteps.npy"), MC_TIMESTEPS)
#np.save(os.path.join(file_path, f"configurations_array{LENGTH}.npy"), configurations)
np.save(os.path.join(file_path, f"energy_per_spin_dimension{LENGTH}.npy"), energy_per_spin)
np.save(os.path.join(file_path, f"magnetisation_per_spin_dimension{LENGTH}.npy"), magnetisation_per_spin)