# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import ising_2d as ising

plt.rcParams.update({'font.size': 18}) # global font parameter for plots

# paths for saving images in report
########################################
report_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Report"
exercise_folder = "FIG/exercise_5_images"
########################################

# Model parameters
LENGTH = 50
CRITICAL_TEMP = 2 / np.log(1 + np.sqrt(2))
BETA_CRITICAL = 1 / CRITICAL_TEMP
temp_mult = (0.25, 0.88, 1.2, 2.5)
TEMP_LIST = np.array(temp_mult) * CRITICAL_TEMP
BETA_LIST = 1 / TEMP_LIST

# Initial configuration

np.random.seed(69420)

SPIN_UP = 1
SPIN_DOWN = -1

initial_configuration = np.random.choice([SPIN_UP, SPIN_DOWN], size=(LENGTH, LENGTH))

np.random.seed(None) # in this way the initial configuration is set always in the same way, but the rest of the evolution no

plt.imshow(initial_configuration, cmap="coolwarm", interpolation="nearest")
cbar = plt.colorbar(ticks=[SPIN_DOWN, SPIN_UP]) 
cbar.set_label("Spin")
plt.title("Initial Spin Configuration")
plt.show()

neighbors_list = ising.neighbors_list_square_pbc(LENGTH) # the neighbor list is configuration independent and can be evaluated at the start of the procedure.   

MC_TIMESTEPS = 5000
TIMESTEPS = LENGTH * LENGTH
beta = 1/ (0.881 * CRITICAL_TEMP)

configurations = np.zeros((len(BETA_LIST), MC_TIMESTEPS, LENGTH, LENGTH))
config = initial_configuration  

for i, beta in enumerate(BETA_LIST):
    for mc_step in range(MC_TIMESTEPS):
        for _ in range(TIMESTEPS):
            config = ising.metropolis_spin_flip_dynamics(config, neighbors_list, LENGTH, beta)
        configurations[i, mc_step] = config  

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
energy_per_spin = np.zeros((len(BETA_LIST), MC_TIMESTEPS))
magnetisation_per_spin = np.zeros((len(BETA_LIST), MC_TIMESTEPS))

for j, beta in enumerate(BETA_LIST):
    for i in range(MC_TIMESTEPS):
        energy_per_spin[j,i] = ising.system_energy(configurations[j,i], neighbors_list) / (LENGTH * LENGTH)
        magnetisation_per_spin[j,i] = np.sum(configurations[j,i]) / (LENGTH * LENGTH)
    
for j, beta in enumerate(BETA_LIST):
    plt.figure(figsize=(10, 10))
    plt.plot(time, energy_per_spin[j], label = f"energy per spin at {1/beta/CRITICAL_TEMP} T_c")
    plt.plot(time, magnetisation_per_spin[j], label = f"magnetisation per spin at {1/beta/CRITICAL_TEMP} T_c")
    plt.xlabel("Time in MC timestep units")
    plt.ylabel("Observables")
    plt.legend(fontsize = 10)
    image_name = f"observables_temp{1/beta/CRITICAL_TEMP:.2f}T_c_dimension{LENGTH}.png"
    entire_path = os.path.join(report_path, exercise_folder, image_name)
    plt.savefig(entire_path)
    plt.close()

# %%
# Data analysis

t_equilibrium = 1000 # LENGTH = 50

energy = np.zeros(len(BETA_LIST))
error_energy = np.zeros(len(BETA_LIST))
magnetisation = np.zeros(len(BETA_LIST))
error_magnetisation = np.zeros(len(BETA_LIST))
heat_capacity = np.zeros(len(BETA_LIST))
error_heat_capacity = np.zeros(len(BETA_LIST))
magnetic_suscept = np.zeros(len(BETA_LIST))
error_magnetic_suscept = np.zeros(len(BETA_LIST))

energy_autocorrelation = np.zeros((len(BETA_LIST), MC_TIMESTEPS))
magnetisation_autocorrelation = np.zeros((len(BETA_LIST), MC_TIMESTEPS))

for j, beta in enumerate(BETA_LIST):
    energy_autocorrelation[j] = ising.normalized_auto_correlation(energy_per_spin[j], 0, MC_TIMESTEPS)
    magnetisation_autocorrelation[j] = ising.normalized_auto_correlation(magnetisation_per_spin, 0, MC_TIMESTEPS)
    energy[j] = ising.mean_value_observable_equilibrium(energy_per_spin[j], t_equilibrium, MC_TIMESTEPS)
    error_energy[j] = ising.error_observable_equilibrium(energy_per_spin[j], t_equilibrium, MC_TIMESTEPS)
    magnetisation[j] = ising.mean_value_observable_equilibrium(magnetisation_per_spin[j], t_equilibrium, MC_TIMESTEPS)
    error_magnetisation[j] = ising.error_observable_equilibrium(magnetisation_per_spin[j], t_equilibrium, MC_TIMESTEPS)  
    heat_capacity[j] = ising.heat_capacity(ising.variance_observable_equilibrium(energy_per_spin[j], t_equilibrium, MC_TIMESTEPS), 1/beta/CRITICAL_TEMP)
    magnetic_suscept[j] = ising.magnetic_susceptibility(LENGTH, beta, ising.variance_observable_equilibrium(magnetisation_per_spin[j], t_equilibrium, MC_TIMESTEPS))
    print("-----------------------------------------------------------------------------------------------------------------------------------")
    print(f"At T = {1/beta/CRITICAL_TEMP} T_c, for a square lattice of side {LENGTH}, energy = {energy[j]} +- {error_energy[j]}")
    print(f"At T = {1/beta/CRITICAL_TEMP} T_c, for a square lattice of side {LENGTH}, magnetisation = {magnetisation[j]} +- {error_magnetisation[j]}")
    print(f"At T = {1/beta/CRITICAL_TEMP} T_c, for a square lattice of side {LENGTH}, heat capacity = {heat_capacity[j]}")
    print(f"At T = {1/beta/CRITICAL_TEMP} T_c, for a square lattice of side {LENGTH}, magnetic susceptibility = {magnetic_suscept[j]}")
    print("-----------------------------------------------------------------------------------------------------------------------------------")

for j, beta in enumerate(BETA_LIST):
    plt.figure(figsize=(10, 10))
    plt.plot(time, energy_autocorrelation[j], label = f"energy autocorrellation at {1/beta/CRITICAL_TEMP} T_c")
    plt.xlabel("Time in MC timestep units")
    plt.ylabel("Energy autocorrelation")
    plt.legend(fontsize = 10)
    plt.show()

# %%
