import os
import numpy as np
import matplotlib.pyplot as plt
import numba

########################################
#report_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Report"
file_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 2/Exercise_12"
#exercise_folder = "FIG/exercise_6_images"
########################################

beta_list = np.load(os.path.join(file_path, "beta_list.npy"))
energy_results = np.load(os.path.join(file_path, "energy_series.npy")) # N = 20

@numba.njit
def partition_function_rescaling(partition_function : np.ndarray) -> np.ndarray:
    min_val = np.min(partition_function)
    max_val = np.max(partition_function)
    scale_factor = 1 / np.sqrt (min_val * max_val)
    return scale_factor * partition_function

@numba.njit
def acceptance_function(new_partition_function, old_partition_function):
    length = len(new_partition_function)
    tmp = 0 
    for k in range(length):
        tmp += (new_partition_function[k] - old_partition_function[k]) / new_partition_function[k]
    return np.sqrt(tmp**2)    

@numba.njit
def partition_function_recursion(beta_list : np.ndarray, energy_list : np.ndarray) -> np.ndarray:
    """
    Naive evaluation of the partition function in the MHM method. 
    Rescaling optimization
    """
    beta_len = len(beta_list)
    partition_function = np.ones(beta_len)
    new_partition_function = np.ones(beta_len)
    total_iterations = energy_list.shape[1]
    acceptance = 1
    threshold = 1e-3
    while acceptance > threshold:
        for k in range(beta_len):
            for energies in energy_list.flatten():
                tmp1 = 0
                for j, beta_j in enumerate(beta_list):
                    exp_weight = np.exp((beta_list[k] - beta_j) * energies )
                    tmp1 += exp_weight / partition_function[j]  
                new_partition_function[k] += 1 / (total_iterations  * tmp1 ) 
        print("iteration complete")          
        acceptance = np.sqrt(np.sum(((new_partition_function - partition_function) / new_partition_function) ** 2))
        #acceptance = acceptance_function(new_partition_function, partition_function)
        print(acceptance)
        partition_function = new_partition_function.copy()
        new_partition_function = partition_function_rescaling(new_partition_function)     
    return new_partition_function

@numba.njit
def partition_function_evaluation(beta_list : np.ndarray, beta_range : np.ndarray,  energy_list : np.ndarray, partition_function_selected_beta : np.ndarray) -> np.ndarray:
    """
    Naive evaluation of the partition function in the MHM method for a beta mesh. 
    """
    new_partition_function = np.ones(len(beta_range))
    total_iterations = energy_list.shape[1]
    for k, beta in enumerate(beta_range):
        for energies in energy_list.flatten():
            tmp1 = 0
            for j, beta_j in enumerate(beta_list):
                exp_weight = np.exp((beta - beta_j) * energies )
                tmp1 += exp_weight / partition_function_selected_beta[j]  
            new_partition_function[k] += 1 / (total_iterations  * tmp1 )            
    return new_partition_function

@numba.njit
def energy_evaluation(beta_list : np.ndarray, beta_range : np.ndarray, energy_list : np.ndarray, partition_function_selected_beta : np.ndarray, general_partition_function : np.ndarray) -> np.ndarray:
    """
    Naive evaluation of the mean energy in the MHM method for a beta mesh.
    """
    observables = np.ones(len(beta_range))
    total_iterations = energy_list.shape[1]
    for k , beta in enumerate(beta_range):
        for energies in energy_list.flatten():
            tmp1 = 0
            for j, beta_j in enumerate(beta_list):
                exp_weight = np.exp((beta - beta_j) * energies )
                tmp1 += exp_weight / partition_function_selected_beta[j]  
            observables[k] += energies / (total_iterations  * tmp1) 
    return observables / general_partition_function

STEP = 0.5
beta_range = np.arange(beta_list[0], beta_list[-1] + STEP, STEP)

part = partition_function_recursion(beta_list, energy_results)
print(part)
gen_part = partition_function_evaluation(beta_list, beta_range, energy_results, part)
print(gen_part)
mean_energy = energy_evaluation(beta_list, beta_range, energy_results, part, gen_part)

plt.plot(beta_range, mean_energy)
plt.show()