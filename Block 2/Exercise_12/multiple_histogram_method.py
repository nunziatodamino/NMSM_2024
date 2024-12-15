import numpy as np
import numba

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
    new_partition_function = np.ones(beta_len)
    total_iterations = energy_list.shape[1]
    acceptance = 1
    threshold = 1e-7
    while acceptance > threshold:
        for k in range(beta_len):
            for energies in energy_list.flatten():
                tmp = 0
                for j, beta in enumerate(beta_list):
                    exp_weight = np.exp((beta_list[k] - beta) * energies )
                    tmp += exp_weight / partition_function[j]
            new_partition_function[k] = 1 / (total_iterations  * tmp )
        print("iteration complete")           
        acceptance = np.sqrt(np.sum(((new_partition_function - partition_function) / new_partition_function) ** 2))
        partition_function = partition_function_rescaling(partition_function)
        print(acceptance)
        partition_function = new_partition_function
    return new_partition_function

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


@numba.njit
def partition_function_evaluation_vectorized(beta_list : np.ndarray, energy_list : np.ndarray, partition_function_selected_beta : np.ndarray) -> np.ndarray:
    """
    Naive evaluation of the partition function in the MHM method for a beta mesh. 
    Rescaling optimization and vectorizing of the operations
    """
    step = 0.05
    beta_range = np.arange(beta_list[0], beta_list[-1] + step, step)
    partition_function = partition_function_selected_beta
    new_partition_function = np.ones(len(beta_range))
    total_iterations = energy_list.shape[1]
    for k, _ in enumerate(beta_range):
        flattened_energy_list = energy_list.flatten()
        tmp1 = np.outer(flattened_energy_list, beta_list[k] - beta_list)
        tmp2 = np.exp(tmp1) * partition_function
        tmp3 = 1 /  np.sum(tmp2, axis=1) # row sum
        new_partition_function[k] = np.sum(tmp3) / total_iterations
    return new_partition_function

@numba.njit
def observable_evaluation_vectorized(observable_list : np.ndarray, beta_list : np.ndarray, energy_list : np.ndarray, partition_function_selected_beta : np.ndarray, general_partition_function : np.ndarray) -> np.ndarray:
    """
    Naive evaluation of the partition function in the MHM method for a beta mesh. 
    Rescaling optimization and vectorizing of the operations
    """
    step = 0.05
    beta_range = np.arange(beta_list[0], beta_list[-1] + step, step)
    partition_function = partition_function_selected_beta
    observables = np.ones(len(beta_range))
    total_iterations = energy_list.shape[1]
    for k, _ in enumerate(beta_range):
        flattened_energy_list = energy_list.flatten()
        flattened_observable_list = observable_list.flatten()
        tmp1 = np.outer(flattened_energy_list, beta_list[k] - beta_list)
        tmp2 = np.exp(tmp1) * partition_function
        tmp3 = np.sum(flattened_observable_list) /  np.sum(tmp2, axis=1) # row sum
        observables[k] = np.sum(tmp3) / (total_iterations * general_partition_function[k])
    return observables