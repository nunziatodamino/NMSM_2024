import numpy as np
import numba

STEP = 0.05

@numba.njit
def partition_function_rescaling(partition_function : np.ndarray) -> np.ndarray:
    min_val = np.min(partition_function)
    max_val = np.max(partition_function)
    scale_factor = 1 / np.sqrt (min_val * max_val)
    return scale_factor * partition_function

@numba.njit
def partition_function_recursion(beta_list : np.ndarray, energy_list : np.ndarray) -> np.ndarray:
    """
    Naive evaluation of the partition function in the MHM method. 
    Rescaling optimization
    """
    beta_len = len(beta_list)
    partition_function = np.ones(beta_len)
    new_partition_function = np.zeros(beta_len)
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
        print(acceptance)
        partition_function = new_partition_function.copy()
        new_partition_function = partition_function_rescaling(new_partition_function)     
    return new_partition_function

@numba.njit
def partition_function_evaluation(beta_list : np.ndarray, step : np.float32,  energy_list : np.ndarray, partition_function_selected_beta : np.ndarray) -> np.ndarray:
    """
    Naive evaluation of the partition function in the MHM method for a beta mesh.
    Exponential correction for stability 
    """
    beta_range = np.arange(beta_list[0], beta_list[-1] + step, step)
    new_partition_function = np.zeros(len(beta_range))
    total_iterations_log = np.log(energy_list.shape[1])
    for k, beta in enumerate(beta_range):
        for energies in energy_list.flatten():
            tmp1 = 0
            tmp2 = np.zeros(len(beta_list))
            for j, beta_j in enumerate(beta_list):
                tmp2[j] = (beta - beta_j) * energies - np.log(partition_function_selected_beta[j]) + total_iterations_log
            tmp3 = max(tmp2)
            tmp1 = np.exp(tmp3) * np.sum(np.exp(tmp2 - tmp3))  
            new_partition_function[k] += 1 / tmp1             
    return new_partition_function

@numba.njit
def energy_evaluation(beta_list : np.ndarray, step : np.float32, energy_list : np.ndarray, partition_function_selected_beta : np.ndarray, general_partition_function : np.ndarray) -> np.ndarray:
    """
    Naive evaluation of the mean energy in the MHM method for a beta mesh.
    """
    beta_range = np.arange(beta_list[0], beta_list[-1] + step, step)
    observables = np.zeros(len(beta_range))
    total_iterations = energy_list.shape[1]
    for k , beta in enumerate(beta_range):
        for energies in energy_list.flatten():
            tmp1 = 0
            for j, beta_j in enumerate(beta_list):
                exp_weight = np.exp((beta - beta_j) * energies )
                tmp1 += exp_weight / partition_function_selected_beta[j]  
            observables[k] += energies / (total_iterations  * tmp1) 
        observables[k] = observables[k] / general_partition_function[k]    
    return observables

@numba.njit
def mean_observable_evaluation(beta_list : np.ndarray, step : np.float32, observable_array : np.ndarray, energy_list : np.ndarray, partition_function_selected_beta : np.ndarray, general_partition_function : np.ndarray) -> np.ndarray:
    """
    Naive evaluation of a mean observable in the MHM method for a beta mesh.
    """
    beta_range = np.arange(beta_list[0], beta_list[-1] + step, step)
    mean_observables = np.zeros(len(beta_range))
    total_iterations = energy_list.shape[1]
    observable_list = observable_array.flatten()
    for k , beta in enumerate(beta_range):
        for i, energies in enumerate(energy_list.flatten()):
            tmp1 = 0
            for j, beta_j in enumerate(beta_list):
                exp_weight = np.exp((beta - beta_j) * energies )
                tmp1 += exp_weight / partition_function_selected_beta[j]  
            mean_observables[k] += observable_list[i] / (total_iterations  * tmp1) 
        mean_observables[k] = mean_observables[k] / general_partition_function[k]    
    return mean_observables

@numba.njit
def partition_function_log_reshifting(partition_function_log : np.ndarray) -> np.ndarray:
    min_val = np.min(partition_function_log)
    max_val = np.max(partition_function_log)
    scale_factor = -0.5 * (min_val + max_val)
    return scale_factor + partition_function_log

@numba.njit
def acceptance_log_criterion(partition_function_log_old : np.ndarray, partition_function_log_new : np.ndarray) -> np.float64:
    a = max(2 * partition_function_log_new)
    b = max(2 * partition_function_log_old)
    c = max(partition_function_log_new + partition_function_log_old)
    gamma = max(a, b, c)
    tmp1 = np.exp(gamma) * np.sum(np.exp(2 * partition_function_log_old - gamma) + np.exp(2 * partition_function_log_new - gamma) - 2 * np.exp(partition_function_log_new + partition_function_log_old - gamma))
    tmp2 = np.exp(b) * np.sum(np.exp(2 * partition_function_log_old - b))
    tmp3 = tmp1 / tmp2 
    log_acceptance = (gamma - b) / 2 + 0.5 * np.log(tmp3)
    return log_acceptance

@numba.njit
def partition_function_log_recursion(beta_list : np.ndarray, energy_list : np.ndarray) -> np.ndarray:
    """
    Numerical stable evaluation of the partition function in the MHM method. 
    Rescaling optimization
    """
    beta_len = len(beta_list)
    partition_function_log = np.zeros(beta_len)
    new_partition_function_log = np.zeros(beta_len)
    total_iterations = energy_list.shape[1]
    acceptance = 1
    threshold = np.log(1e-7)
    log_total_iterations = np.log(total_iterations)
    while acceptance > threshold:
        for k in range(beta_len):
            for i, energies in enumerate(energy_list.flatten()):
                tmp1 = np.zeros(len(energy_list.flatten()))
                tmp2 = np.zeros(beta_len) 
                tmp3 = 0
                for j, beta in enumerate(beta_list):
                    tmp2[j] = (beta_list[k] - beta) * energies + log_total_iterations - partition_function_log[j] #\lambda_ijk
                tmp1[i] = max(tmp2) #\lambda_ij^*k
                tmp3 = np.sum(np.exp(tmp2 - tmp1[i]))
            max_term = max(tmp1)
            tmp4 = np.sum(np.exp(-tmp1 + max_term) / tmp3)    
            new_partition_function_log[k] = - max_term + np.log(tmp4)         
        print("iteration complete")           
        acceptance = acceptance_log_criterion(partition_function_log, new_partition_function_log)
        new_partition_function_log = partition_function_log_reshifting(new_partition_function_log)
        print(acceptance)
        partition_function_log = new_partition_function_log
    return new_partition_function_log

@numba.njit
def partition_function_recursion_vectorized(beta_list : np.ndarray, energy_list : np.ndarray) -> np.ndarray:
    """
    Naive evaluation of the partition function in the MHM method. 
    Rescaling optimization and vectorizing of the operations
    """
    beta_len = len(beta_list)
    partition_function = np.ones(beta_len)
    new_partition_function = np.ones(beta_len)
    total_iterations = energy_list.shape[1]
    acceptance = 1
    threshold = 1e-7
    while acceptance > threshold:
        for k in range(beta_len):
            flattened_energy_list = energy_list.flatten()
            tmp1 = np.outer(flattened_energy_list, beta_list[k] - beta_list)
            tmp2 = np.exp(tmp1) * partition_function
            tmp3 = 1 /  np.sum(tmp2, axis=1) # row sum
            new_partition_function[k] = np.sum(tmp3) / total_iterations
        print("iteration complete")           
        acceptance = np.sqrt(np.sum(((new_partition_function - partition_function) / new_partition_function) ** 2))
        partition_function = partition_function_rescaling(partition_function)
        print(acceptance)
        partition_function = new_partition_function
    return new_partition_function


