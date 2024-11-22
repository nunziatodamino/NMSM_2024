
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit
from statsmodels.graphics.tsaplots import plot_acf
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

#energy_per_spin = np.load(os.path.join(file_path, "energy_per_spin_dimension25.npy"))
#magnetisation_per_spin = np.load(os.path.join(file_path, "magnetisation_per_spin_dimension25.npy"))
#LENGTH = 25

#energy_per_spin = np.load(os.path.join(file_path, "energy_per_spin_dimension50.npy"))
#magnetisation_per_spin = np.load(os.path.join(file_path, "magnetisation_per_spin_dimension50.npy"))
#LENGTH = 50

energy_per_spin = np.load(os.path.join(file_path, "energy_per_spin_dimension100.npy"))
magnetisation_per_spin = np.load(os.path.join(file_path, "magnetisation_per_spin_dimension100.npy"))
LENGTH = 100

#t_equilibrium_list = [ 2000, 2000, 2500, 50 ] # LENGTH = 25 
#t_equilibrium_list = [ 500, 1700, 8000, 50 ] # LENGTH = 50 
t_equilibrium_list = [ 10000, 10000, 10000, 10000 ] # LENGTH = 100 

energy = np.zeros(len(BETA_LIST))
error_energy = np.zeros(len(BETA_LIST))
magnetisation = np.zeros(len(BETA_LIST))
error_magnetisation = np.zeros(len(BETA_LIST))
heat_capacity = np.zeros(len(BETA_LIST))
error_heat_capacity = np.zeros(len(BETA_LIST))
magnetic_suscept = np.zeros(len(BETA_LIST))
error_magnetic_suscept = np.zeros(len(BETA_LIST))

tau_fit_energy_error = np.zeros(len(BETA_LIST))
tau_fit_magn_error = np.zeros(len(BETA_LIST))

energy_autocorrelation = np.empty(len(BETA_LIST), dtype=object)
magnetisation_autocorrelation = np.empty(len(BETA_LIST), dtype=object)
tau_energy_autocorrelation = np.zeros(len(BETA_LIST))
tau_magnetisation_autocorrelation = np.zeros(len(BETA_LIST))

# autocorrelation study

def exponential_decay(t, tau):
    return np.exp(-t / tau)

block_size = 10000

total_energy = LENGTH * LENGTH * energy_per_spin
total_magnetisation = LENGTH * LENGTH * magnetisation_per_spin

for j, beta in enumerate(BETA_LIST):
        
    energy_autocorrelation[j] = np.zeros(MC_TIMESTEPS-t_equilibrium_list[j])
    magnetisation_autocorrelation[j] = np.zeros(MC_TIMESTEPS-t_equilibrium_list[j])
        
    energy_autocorrelation[j] = acf(energy_per_spin[j][t_equilibrium_list[j]:], nlags = MC_TIMESTEPS-t_equilibrium_list[j])
    magnetisation_autocorrelation[j] = acf(magnetisation_per_spin[j][t_equilibrium_list[j]:], nlags = MC_TIMESTEPS-t_equilibrium_list[j])
        
    time_lags = np.arange(MC_TIMESTEPS-t_equilibrium_list[j])
    fit_energy, cov_energy  = curve_fit(exponential_decay, time_lags, energy_autocorrelation[j], p0=(10,))
    tau_energy_autocorrelation[j] = fit_energy[0]
    tau_fit_energy_error[j] = np.sqrt(np.diag(cov_energy))[0]
        
    fit_magn, cov_magn = curve_fit(exponential_decay, time_lags, magnetisation_autocorrelation[j], p0=(10,))
    tau_magnetisation_autocorrelation[j] = fit_magn[0]
    tau_fit_magn_error[j] = np.sqrt(np.diag(cov_magn))[0]

    energy[j] = ising.mean_value_observable_equilibrium(energy_per_spin[j], t_equilibrium_list[j], MC_TIMESTEPS)
    error_energy[j] = ising.error_observable_corr_equilibrium(energy_per_spin[j], t_equilibrium_list[j], MC_TIMESTEPS, tau_energy_autocorrelation[j])
        
    magnetisation[j] = ising.mean_value_observable_equilibrium(magnetisation_per_spin[j], t_equilibrium_list[j], MC_TIMESTEPS)
    error_magnetisation[j] = ising.error_observable_corr_equilibrium(magnetisation_per_spin[j], t_equilibrium_list[j], MC_TIMESTEPS, tau_magnetisation_autocorrelation[j])  

    heat_capacity[j] , error_heat_capacity[j] = ising.block_averaging_heat_capacity(total_energy[j], t_equilibrium_list[j], MC_TIMESTEPS, block_size, 1/beta) 
    magnetic_suscept[j] , error_magnetic_suscept[j]= ising.block_averaging_magnetic_susc(total_magnetisation[j], t_equilibrium_list[j], MC_TIMESTEPS, block_size, beta) 

    TOTAL_SPINS = LENGTH * LENGTH
    heat_capacity[j] /= TOTAL_SPINS
    error_heat_capacity[j] /= TOTAL_SPINS
    magnetic_suscept[j] /= TOTAL_SPINS
    error_magnetic_suscept[j] /= TOTAL_SPINS

    # LaTeX table print
    print("\\hline")
    print(f"${1/beta/CRITICAL_TEMP:.6f} T_c$ & "
          f"${tau_energy_autocorrelation[j]:.6f} \\pm {tau_fit_energy_error[j]:.6f}$ & "
          f"${tau_magnetisation_autocorrelation[j]:.6f} \\pm {tau_fit_magn_error[j]:.6f}$ & "
          f"${energy[j]:.6f} \\pm {error_energy[j]:.6f}$ & "
          f"${magnetisation[j]:.6f} \\pm {error_magnetisation[j]:.6f}$ & "
          f"${heat_capacity[j]:.6f} \\pm {error_heat_capacity[j]:.6f}$ & "
          f"${magnetic_suscept[j]:.6f} \\pm {error_magnetic_suscept[j]:.6f}$ \\\\")


for j, beta in enumerate(BETA_LIST):
    time_lags = np.arange(MC_TIMESTEPS-t_equilibrium_list[j])
    plt.figure(figsize=(10, 10))
    plt.plot(time_lags, energy_autocorrelation[j], label = f"energy autocorrelation at {1/beta/CRITICAL_TEMP} T_c")
    plt.plot(time_lags, magnetisation_autocorrelation[j], label = f"magnetisation autocorrelation at {1/beta/CRITICAL_TEMP} T_c")
    plt.xlabel(f"Time in  MC steps")
    plt.ylabel("Autocorrelation")
    plt.legend(fontsize = 10)
    #plt.yscale('log')
    image_name = f"observables_autocorrelation_temp{1/beta/CRITICAL_TEMP:.6f}T_c_dimension{LENGTH}.png"
    entire_path = os.path.join(report_path, exercise_folder, image_name)
    plt.savefig(entire_path)
    plt.close()
    #plt.show()

data_dict = {
    "Beta": BETA_LIST,
    "Energy": energy,
    "Error in Energy": error_energy,
    "Tau Energy Autocorrelation": tau_energy_autocorrelation,
    "Magnetisation": magnetisation,
    "Error in Magnetisation": error_magnetisation,
    "Tau Magnetisation Autocorrelation": tau_magnetisation_autocorrelation,
    "Heat Capacity per spin": heat_capacity,
    "Error in Heat Capacity": error_heat_capacity,
    "Magnetic Susceptibility per spin": magnetic_suscept,
    "Error in Magnetic Susceptibility": error_magnetic_suscept
}
