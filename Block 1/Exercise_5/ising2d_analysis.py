# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit
import ising_2d as ising

plt.rcParams.update({'font.size': 18}) # global font parameter for plots

# paths for saving images in report
########################################
report_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Report"
file_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Exercise_5"
exercise_folder = "FIG/exercise_5_images"
########################################

CRITICAL_TEMP = np.load(os.path.join(file_path, "critical_temp.npy")) 
BETA_LIST = np.load(os.path.join(file_path, "beta_list.npy")) 
MC_TIMESTEPS = np.load(os.path.join(file_path, "mc_timesteps.npy"))

energy_per_spin = np.load(os.path.join(file_path, "energy_per_spin_dimension50.npy"))
magnetisation_per_spin = np.load(os.path.join(file_path, "magnetisation_per_spin_dimension50.npy"))
LENGTH = 50

#energy_per_spin = np.load("energy_per_spin_dimension25.npy")
#magnetisation_per_spin = np.load("magnetisation_per_spin_dimension25.npy")
#LENGTH = 25

# stationarity check
#for i, beta in enumerate(BETA_LIST):
#    print(f"Is energy weakly stationary T = {1/beta/CRITICAL_TEMP} T_c ? : {ising.is_weakly_stationary(energy_per_spin[i][3000:])}")
#    print(f"Is magnetization weakly stationary at T = {1/beta/CRITICAL_TEMP} T_c ? : {ising.is_weakly_stationary(magnetisation_per_spin[i][3000:])}")
#    print("-----------------------------------------------------------------------------------------------------------")

t_equilibrium = 2000 # LENGTH = 50
lags_list = [100000, 100000, 1000000, 100000] # LENGTH = 50

energy = np.zeros(len(BETA_LIST))
error_energy = np.zeros(len(BETA_LIST))
magnetisation = np.zeros(len(BETA_LIST))
error_magnetisation = np.zeros(len(BETA_LIST))
heat_capacity = np.zeros(len(BETA_LIST))
error_heat_capacity = np.zeros(len(BETA_LIST))
magnetic_suscept = np.zeros(len(BETA_LIST))
error_magnetic_suscept = np.zeros(len(BETA_LIST))

energy_autocorrelation = np.empty(len(BETA_LIST), dtype=object)
magnetisation_autocorrelation = np.empty(len(BETA_LIST), dtype=object)
tau_energy_autocorrelation = np.zeros(len(BETA_LIST))
tau_magnetisation_autocorrelation = np.zeros(len(BETA_LIST))
lags_cutoff_energy= np.zeros(len(BETA_LIST))
lags_cutoff_magn= np.zeros(len(BETA_LIST))

# autocorrelation study

def set_cutoff(exponential):
    counter = 0
    while exponential[counter] > 1/np.e:
        counter+=1
    return int(counter)

def exponential_decay(t, tau):
    return np.exp(-t / tau)

for j, beta in enumerate(BETA_LIST):
    #energy_autocorrelation[j] = np.zeros(lags_list[j])
    #magnetisation_autocorrelation[j] = np.zeros(lags_list[j])
    energy_autocorrelation[j] = np.zeros(MC_TIMESTEPS)
    magnetisation_autocorrelation[j] = np.zeros(MC_TIMESTEPS)
    energy_autocorrelation[j] = acf(energy_per_spin[j], nlags = MC_TIMESTEPS)
    magnetisation_autocorrelation[j] = acf(magnetisation_per_spin[j], nlags = MC_TIMESTEPS)
    # set lag cutoff
    lags_cutoff_energy[j]=set_cutoff(energy_autocorrelation[j])
    lags_cutoff_magn[j]=set_cutoff(magnetisation_autocorrelation[j])
    tau_energy_autocorrelation[j] = 1/2 + np.sum(energy_autocorrelation[j][:int(lags_cutoff_energy[j])]) # integrated autocorrelation time definition + cutoff
    tau_magnetisation_autocorrelation[j] = 1/2 + np.sum(magnetisation_autocorrelation[j][:int(lags_cutoff_magn[j])])
    time_lags = np.arange(MC_TIMESTEPS)
    popt_energy, _ = curve_fit(exponential_decay, time_lags, energy_autocorrelation[j], p0=(10,))
    tau_fit_energy = popt_energy[0]
    popt_magn, _ = curve_fit(exponential_decay, time_lags, magnetisation_autocorrelation[j], p0=(10,))
    tau_fit_magnetisation = popt_magn[0]
    
    print(f"At T = {1/beta/CRITICAL_TEMP} T_c, for a square lattice of side {LENGTH}, tau energy autocorrelation = {tau_energy_autocorrelation[j] } (in MC timesteps)")
    print(f"At T = {1/beta/CRITICAL_TEMP} T_c, for a square lattice of side {LENGTH}, tau magnetisation autocorrelation = {tau_magnetisation_autocorrelation[j] } (in MC timesteps) ")
    print(f"  Tau energy (fit)       = {tau_fit_energy} (in MC timesteps)")
    print(f"  Tau magnetisation (fit)       = {tau_fit_magnetisation} (in MC timesteps)")
    print("-----------------------------------------------------------------------------------------------------------------------------------")

for j, beta in enumerate(BETA_LIST):
    time = np.arange(0, MC_TIMESTEPS, 1)
    plt.figure(figsize=(10, 10))
    plt.plot(time, energy_autocorrelation[j], label = f"energy autocorrelation at {1/beta/CRITICAL_TEMP} T_c")
    plt.plot(time, magnetisation_autocorrelation[j], label = f"magnetisation autocorrelation at {1/beta/CRITICAL_TEMP} T_c")
    plt.xlabel(f"Time in  MC steps")
    plt.ylabel("Autocorrelation")
    plt.legend(fontsize = 10)
    #plt.yscale('log')
    image_name = f"observables_autocorrelation_temp{1/beta/CRITICAL_TEMP:.2f}T_c_dimension{LENGTH}.png"
    entire_path = os.path.join(report_path, exercise_folder, image_name)
    plt.savefig(entire_path)
    plt.close()
    #plt.show()

#for j, beta in enumerate(BETA_LIST):
#    energy[j] = ising.mean_value_observable_equilibrium(energy_per_spin[j], t_equilibrium, MC_TIMESTEPS)
#    error_energy[j] = ising.error_observable_equilibrium(energy_per_spin[j], t_equilibrium, MC_TIMESTEPS)
#    magnetisation[j] = ising.mean_value_observable_equilibrium(magnetisation_per_spin[j], t_equilibrium, MC_TIMESTEPS)
#    error_magnetisation[j] = ising.error_observable_equilibrium(magnetisation_per_spin[j], t_equilibrium, MC_TIMESTEPS)  
#    heat_capacity[j] = ising.heat_capacity(ising.variance_observable_equilibrium(energy_per_spin[j], t_equilibrium, MC_TIMESTEPS), 1/beta/CRITICAL_TEMP)
#    magnetic_suscept[j] = ising.magnetic_susceptibility(LENGTH, beta, ising.variance_observable_equilibrium(magnetisation_per_spin[j], t_equilibrium, MC_TIMESTEPS))
#    print(f"At T = {1/beta/CRITICAL_TEMP} T_c, for a square lattice of side {LENGTH}, energy = {energy[j]} +- {error_energy[j]}")
#    print(f"At T = {1/beta/CRITICAL_TEMP} T_c, for a square lattice of side {LENGTH}, magnetisation = {magnetisation[j]} +- {error_magnetisation[j]}")
#    print(f"At T = {1/beta/CRITICAL_TEMP} T_c, for a square lattice of side {LENGTH}, heat capacity = {heat_capacity[j]}")
#    print(f"At T = {1/beta/CRITICAL_TEMP} T_c, for a square lattice of side {LENGTH}, magnetic susceptibility = {magnetic_suscept[j]}")

# %%
