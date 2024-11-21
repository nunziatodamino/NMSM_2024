import os
import numpy as np
#import pandas as pd                        # decomment if you want to see data using the dataframe
#from tabulate import tabulate 
from statsmodels.tsa.stattools import acf
from scipy.optimize import curve_fit
import polymer as polymer

########################################
report_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Report"
file_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Exercise_6"
exercise_folder = "FIG/exercise_6_images"
########################################

#t_equilibrium = 60000
#time_max = np.load(os.path.join(file_path, "time_max.npy"))
beta_list = np.load(os.path.join(file_path, "beta_list.npy"))

#energy_evolution_results = np.load(os.path.join(file_path, "energy_series.npy"))
#ee2_results = np.load(os.path.join(file_path, "ee2_series.npy"))
#end_height_results = np.load(os.path.join(file_path, "end2height_series.npy"))
#gyr_rad_results = np.load(os.path.join(file_path, "gyr_radius_series.npy"))

time_max = 1000000
t_equilibrium = 300000
energy_evolution_results = np.load(os.path.join(file_path, "mmc_energy_series.npy"))
ee2_results = np.load(os.path.join(file_path, "mmc_ee2_series.npy"))
end_height_results = np.load(os.path.join(file_path, "mmc_end2height_series.npy"))
gyr_rad_results = np.load(os.path.join(file_path, "mmc_gyr_radius_series.npy"))

energy_autocorr=np.zeros((len(beta_list), time_max - t_equilibrium))
end2end_autocorr=np.zeros((len(beta_list), time_max - t_equilibrium))
end_height_autocorr=np.zeros((len(beta_list), time_max - t_equilibrium))
gyr_rad_autocorr=np.zeros((len(beta_list), time_max - t_equilibrium))

tau_energy_autocorr = np.zeros(len(beta_list))
tau_ee2_autocorr = np.zeros(len(beta_list))
tau_endheight_autocorr = np.zeros(len(beta_list))
tau_gyr_rad_autocorr = np.zeros(len(beta_list))

mean_energy = np.zeros(len(beta_list))
mean_ee2 = np.zeros(len(beta_list))
mean_end_height = np.zeros(len(beta_list))
mean_gyr_rad = np.zeros(len(beta_list))

energy_variance = np.zeros(len(beta_list))
energy_variance_corr = np.zeros(len(beta_list))

error_energy = np.zeros(len(beta_list))
error_ee2 = np.zeros(len(beta_list))
error_end_height = np.zeros(len(beta_list))
error_gyr_rad = np.zeros(len(beta_list))

tau_fit_energy_error = np.zeros(len(beta_list))
tau_fit_endheight_error = np.zeros(len(beta_list))
tau_fit_ee2_error = np.zeros(len(beta_list))
tau_fit_gyr_rad_error = np.zeros(len(beta_list))


def exponential_decay(t, tau):
    return np.exp(-t / tau)

time_lags = np.arange(time_max - t_equilibrium)

for i, beta in enumerate(beta_list):
    
    energy_autocorr[i] = acf(energy_evolution_results[i][t_equilibrium:], nlags = time_max - t_equilibrium)
    end2end_autocorr[i] = acf(ee2_results[i][t_equilibrium:], nlags = time_max - t_equilibrium)
    end_height_autocorr[i] = acf(end_height_results[i][t_equilibrium:], nlags = time_max - t_equilibrium)
    gyr_rad_autocorr[i] = acf(gyr_rad_results[i][t_equilibrium:], nlags = time_max - t_equilibrium)

    fit_energy, cov_energy  = curve_fit(exponential_decay, time_lags, energy_autocorr[i], p0=(10,))
    fit_ee2, cov_ee2 = curve_fit(exponential_decay, time_lags, end2end_autocorr[i], p0=(10,))
    fit_endheight, cov_endheight = curve_fit(exponential_decay, time_lags, end_height_autocorr[i], p0=(10,))
    fit_gyr_rad, cov_gyr_rad = curve_fit(exponential_decay, time_lags, gyr_rad_autocorr[i], p0=(10,))
    
    tau_energy_autocorr[i] = fit_energy[0]
    tau_ee2_autocorr[i] = fit_ee2[0]
    tau_endheight_autocorr[i] = fit_endheight[0]
    tau_gyr_rad_autocorr[i] = fit_gyr_rad[0]
    
    tau_fit_energy_error[i] = np.sqrt(np.diag(cov_energy))[0]    
    tau_fit_ee2_error[i] = np.sqrt(np.diag(cov_ee2))[0]
    tau_fit_endheight_error[i] = np.sqrt(np.diag(cov_endheight))[0]
    tau_fit_gyr_rad_error[i] = np.sqrt(np.diag(cov_gyr_rad))[0]

    mean_energy[i] = polymer.mean_value_observable_equilibrium(energy_evolution_results[i], t_equilibrium, time_max)
    mean_ee2[i] = polymer.mean_value_observable_equilibrium(ee2_results[i], t_equilibrium, time_max)
    mean_end_height[i] = polymer.mean_value_observable_equilibrium(end_height_results[i], t_equilibrium, time_max)
    mean_gyr_rad[i] = polymer.mean_value_observable_equilibrium(gyr_rad_results[i], t_equilibrium, time_max)

    energy_variance[i] = polymer.variance_observable_equilibrium(energy_evolution_results[i], t_equilibrium, time_max)
    energy_variance_corr[i] = polymer.variance_observable_corr_equilibrium(energy_evolution_results[i], t_equilibrium, time_max, tau_energy_autocorr[i])

    error_energy[i] = polymer.error_observable_corr_equilibrium(energy_evolution_results[i], t_equilibrium, time_max, tau_energy_autocorr[i])
    error_ee2[i] = polymer.error_observable_corr_equilibrium(ee2_results[i], t_equilibrium, time_max, tau_ee2_autocorr[i])
    error_end_height[i] = polymer.error_observable_corr_equilibrium(end_height_results[i], t_equilibrium, time_max, tau_endheight_autocorr[i])
    error_gyr_rad[i] = polymer.error_observable_corr_equilibrium(gyr_rad_results[i], t_equilibrium, time_max, tau_gyr_rad_autocorr[i])
    
    #print(f"at beta = {beta}, energy_var = {energy_variance[i]}, energy_var_corr = {energy_variance_corr[i]} ")
    # LaTeX table print
    print("\\hline")
    print(f"${beta:.2f}$ & "
          f"${tau_energy_autocorr[i]:.2f} \\pm {tau_fit_energy_error[i]:.2f}$ & "
          f"${tau_ee2_autocorr[i]:.2f} \\pm {tau_fit_ee2_error[i]:.2f}$ & "
          f"${tau_endheight_autocorr[i]:.2f} \\pm {tau_fit_endheight_error[i]:.2f}$ & "
          f"${tau_gyr_rad_autocorr[i]:.2f} \\pm {tau_fit_gyr_rad_error[i]:.2f}$ & "
          f"${mean_energy[i]:.2f} \\pm {error_energy[i]:.2f}$ & "
          f"${mean_ee2[i]:.2f} \\pm {error_ee2[i]:.2f}$ & "
          f"${mean_end_height[i]:.2f} \\pm {error_end_height[i]:.2f}$ & "
          f"${mean_gyr_rad[i]:.2f} \\pm {error_gyr_rad[i]:.2f}$ \\\\")

#data = {
#    "$\\beta$": [f"${beta:.2f}$" for beta in beta_list],
#    "$\\tau_{\\text{Energy}}$": [f"${tau:.2f} \\pm {err:.2f}$" for tau, err in zip(tau_energy_autocorr, tau_fit_energy_error)],
#    "$\\tau_{ee2}$": [f"${tau:.2f} \\pm {err:.2f}$" for tau, err in zip(tau_ee2_autocorr, tau_fit_ee2_error)],
#    "$\\tau_{\\text{End Height}}$": [f"${tau:.2f} \\pm {err:.2f}$" for tau, err in zip(tau_endheight_autocorr, tau_fit_endheight_error)],
#    "$\\tau_{\\text{Gyr Rad}}$": [f"${tau:.2f} \\pm {err:.2f}$" for tau, err in zip(tau_gyr_rad_autocorr, tau_fit_gyr_rad_error)],
#    "$\\text{Energy}$": [f"${mean:.2f} \\pm {err:.2f}$" for mean, err in zip(mean_energy, error_energy)],
#    "$\\text{ee2}$": [f"${mean:.2f} \\pm {err:.2f}$" for mean, err in zip(mean_ee2, error_ee2)],
#    "$\\text{End Height}$": [f"${mean:.2f} \\pm {err:.2f}$" for mean, err in zip(mean_end_height, error_end_height)],
#    "$\\text{Gyr Rad}$": [f"${mean:.2f} \\pm {err:.2f}$" for mean, err in zip(mean_gyr_rad, error_gyr_rad)],
#}

#df = pd.DataFrame(data)
#print(tabulate(df, headers='keys', tablefmt='grid', showindex=False))