import numpy as np
from numba import njit
from statsmodels.tsa.stattools import acf

J = 1

def neighbors_list_square_pbc(length):
    """
    Evaluates the neighbors list of the square lattice of side length respecting the periodic boundary conditions.
    The hashtable is stored in a dictionary (more pythonic) 
    """
    neighbors_list = {}
    for i in range(length):
        for j in range(length):
            neighbors = []
            # Apply periodic boundary conditions (PBC) with modulo operator
            neighbors.append(((i - 1) % length, j))  
            neighbors.append(((i + 1) % length, j))  
            neighbors.append((i, (j - 1) % length))  
            neighbors.append((i, (j + 1) % length))        
            neighbors_list[(i, j)] = neighbors           
    return neighbors_list

@njit
def neighbors_list_square_pbc_opt(length):
    """
    Evaluates the neighbors list of the square lattice of side length respecting the periodic boundary conditions.
    The hashtable is stored in an array (more efficient for further optimations) 
    """
    neighbors_list = np.zeros((length, length, 4, 2), dtype=np.int32)
    for i in range(length):
        for j in range(length):
            neighbors_list[i, j, 0] = ((i - 1) % length, j)  
            neighbors_list[i, j, 1] = ((i + 1) % length, j)  
            neighbors_list[i, j, 2] = (i, (j - 1) % length)  
            neighbors_list[i, j, 3] = (i, (j + 1) % length)  
    return neighbors_list

def system_energy(configuration, neighbors_list):
    '''
    Evaluates the system energy using the configuration of spin and the neighbour list.
    Divides by 2 to account for overcounting.
    '''
    energy = 0 
    for (i, j), neighbors in neighbors_list.items():
        for neighbor in neighbors:
            neighbor_i, neighbor_j = neighbor
            energy -= J * configuration[i][j] * configuration[neighbor_i][neighbor_j]
    return energy / 2

@njit
def system_energy_opt(configuration, neighbors_list, length):
    '''
    Optimized version for numba.
    Evaluates the system energy using the configuration of spin and the neighbour list.
    Divides by 2 to account for overcounting.
    '''
    energy = 0.0
    for i in range(length):
        for j in range(length):
            for k in range(4):  # 4 neighbors
                neighbor = neighbors_list[i, j, k]
                neighbor_i, neighbor_j = neighbor[0], neighbor[1]
                energy -= J * configuration[i, j] * configuration[neighbor_i, neighbor_j]
    return energy / 2.0

@njit
def metropolis_spin_flip_dynamics_opt(old_configuration, neighbors_list, length, beta):
    '''
    Optimized version for numba
    Metropolis filter for one spin flip
    '''
    i = np.random.randint(0, length)
    j = np.random.randint(0, length)

    neighbor_sum = 0
    for k in range(4):  # 4 neighbors
        neighbor = neighbors_list[i, j, k]
        neighbor_sum += old_configuration[neighbor[0], neighbor[1]]

    delta_energy = 2 * J * old_configuration[i, j] * neighbor_sum
    new_configuration = np.copy(old_configuration)
    if delta_energy <= 0 or np.random.random() <= np.exp(-beta * delta_energy):
        new_configuration[i, j] = -old_configuration[i, j]  # Spin flip
    return new_configuration


def metropolis_spin_flip_dynamics(old_configuration, neighbors_list, length, beta):
    '''
    Metropolis filter for one spin flip
    '''
    i = np.random.randint(0, length)
    j = np.random.randint(0, length)
    neighbor_sum = sum(old_configuration[neighbor] for neighbor in neighbors_list[(i, j)])
    delta_energy = 2 * J * old_configuration[i,j] * neighbor_sum
    new_configuration= old_configuration.copy()
    new_configuration[i,j] = -old_configuration[i,j] # spin flip
    if delta_energy <= 0: return new_configuration
    elif np.random.random() <= np.exp(- beta * delta_energy ) : return new_configuration
    return old_configuration   

def glauber_spin_flip_dynamics(old_configuration, neighbors_list, length, beta):
    '''
    Glauber filter for one spin flip (optional)
    '''
    i = np.random.randint(0, length)
    j = np.random.randint(0, length)
    neighbor_sum = sum(old_configuration[neighbor] for neighbor in neighbors_list[(i, j)])
    delta_energy = 2 * J * old_configuration[i,j] * neighbor_sum
    new_configuration= old_configuration.copy()
    new_configuration[i,j] = -old_configuration[i,j] # spin flip
    if np.random.random() <= 1 / (1 + np.exp( beta * delta_energy )) : return new_configuration
    return old_configuration

def mean_value_observable_equilibrium(observable, t_equilibrium, t_max):
    '''
    Evaluates the mean of an observable time series long t_max , discarding the first t_equilibrium values
    '''
    return 1/(t_max - t_equilibrium) * np.sum(observable[t_equilibrium : t_max])

def variance_observable_equilibrium(observable, t_equilibrium, t_max):
    '''
    Evaluates the variance of an observable time series long t_max , discarding the first t_equilibrium values
    '''
    tmp = 1/(t_max - t_equilibrium) * np.sum(observable[t_equilibrium : t_max]**2)
    return tmp - mean_value_observable_equilibrium(observable, t_equilibrium, t_max)**2

def error_observable_equilibrium(observable, t_equilibrium, t_max):
    '''
    Evaluates the error of an observable time series long t_max , discarding the first t_equilibrium values
    '''
    return np.sqrt(variance_observable_equilibrium(observable, t_equilibrium, t_max) / (t_max - t_equilibrium))

def variance_observable_corr_equilibrium(observable, t_equilibrium, t_max, tau):
    '''
    Evaluates the variance of an observable time series long t_max , discarding the first t_equilibrium values,
    considering the data are correlated, so the correlation time tau must be supplied.
    '''
    tmp = (2 * tau)/(t_max - t_equilibrium) * np.sum(observable[t_equilibrium : t_max]**2)
    return tmp - mean_value_observable_equilibrium(observable, t_equilibrium, t_max)**2

def error_observable_corr_equilibrium(observable, t_equilibrium, t_max, tau):
    '''
    Evaluates the error of an observable time series long t_max , discarding the first t_equilibrium values,
    considering the data are correlated, so the correlation time tau must be supplied.
    '''
    return np.sqrt(variance_observable_corr_equilibrium(observable, t_equilibrium, t_max, tau) / (t_max - t_equilibrium))

def auto_correlation_opt(distribution, time_max):
    '''
    ATTENTION: NOT WORKING
    Optimized autocorrelation function to mitigate the effect of the numerical instability, but still presents it.
    '''    
    auto_corr = np.zeros(time_max)
    shifted_distribution = distribution.copy()
    shifted_distribution[1:] = distribution[:-1] # creates shift by 1
    for t in range(time_max):
        counter = 1 
        shifted_distribution = distribution.copy()
        shifted_distribution[1:] = distribution[:-counter] # creates shift by conunter
        tmp_sum_1 = distribution * shifted_distribution
        c1 = np.sum(tmp_sum_1[counter:])
        c2 = np.sum(distribution[counter:])
        c3 = np.sum(shifted_distribution[counter:]) 
        prefactor = 1 / (time_max - counter)
        auto_corr[t] = prefactor * ( c1 - c2 * c3 ) 
        counter +=1    
    return auto_corr/auto_corr[0]  

def is_weakly_stationary(observable, window_size=4000, acc_threshold=0.1):
    '''
    ATTENTION: PROOF OF CONCEPT, ADF TEST SHOULD BE USED
    Given an observable time series returns True is the process is weakly stationary.
    This function is a proof of concept because it can never work : I think the relative change can be of the
    order of the machine_eps or the floating rounding error sometimes, leading to numerical instability and default
    False even when the process is weakly stationary. 
    '''
    means = []
    variances = []
    autocorrelations = []
    
    for i in range(0, len(observable), window_size):
        chunk = observable[i:i+window_size]
        if len(chunk) < window_size: 
            continue
        means.append(np.mean(chunk))
        variances.append(np.var(chunk))
        autocorrelations.append(np.mean(acf(chunk, nlags=10)))  
        
    def relative_change(data):
        return (max(data) - min(data)) / np.mean(data)  

    mean_change = relative_change(means)
    variance_change = relative_change(variances)
    acf_change = relative_change(autocorrelations)

    if mean_change < acc_threshold and variance_change < acc_threshold and acf_change < acc_threshold:
        return True
    return False


def block_averaging_heat_capacity(energies, t_equilibrium, t_max , block_size, temperature):
    '''
    Evaluates the heat capacity via the block averaging technique. Takes the energy time series long t_max,
    discardes the first t_equilibrium values, and then performs the block averaging where each bloch has lenght 
    block_size.
    The temperature value is used to evaluate the heat capacity.
    '''
    energies=energies[t_equilibrium:t_max]
    n_data = len(energies)
    n_blocks = n_data // block_size
    energies = energies[:n_blocks * block_size] # trim if something goes wrong
    blocks = energies.reshape(n_blocks, block_size)
    avg_e = np.mean(blocks, axis=1)
    avg_e2 = np.mean(blocks**2, axis=1)
    cv_blocks = (avg_e2 - avg_e**2) / temperature**2
    mean_cv = np.mean(cv_blocks)
    std_cv = np.std(cv_blocks, ddof=1)  
    error_cv = std_cv / np.sqrt(n_blocks)
    return mean_cv, error_cv

def block_averaging_magnetic_susc(magnetisation, t_equilibrium, t_max , block_size, beta):
    '''
    Evaluates the magnetic susceptibility via the block averaging technique. Takes the magnetisation time series long t_max,
    discardes the first t_equilibrium values, and then performs the block averaging where each bloch has lenght 
    block_size.
    The beta value is used to evaluate the magnetic susceptibility.
    '''
    magnetisation=magnetisation[t_equilibrium:t_max]
    n_data = len(magnetisation)
    n_blocks = n_data // block_size
    magnetisation = magnetisation[:n_blocks * block_size] # trim if something goes wrong 
    blocks = magnetisation.reshape(n_blocks, block_size)
    avg_m = np.mean(blocks, axis=1)
    avg_m2 = np.mean(blocks**2, axis=1)
    chi_m_blocks = (avg_m2 - avg_m**2) * beta
    mean_chi_m = np.mean(chi_m_blocks)
    std_chi_m = np.std(chi_m_blocks, ddof=1)  
    error_chi_m = std_chi_m / np.sqrt(n_blocks)
    return mean_chi_m, error_chi_m

###################################
# Optional - GIF
###################################

# from PIL import Image, ImageDraw

# frames = []
# output_size = (350, 350)
# for i, frame in enumerate(configurations):
#     img = Image.fromarray(frame.astype(np.uint8)).convert("L") 
#     img = img.resize(output_size, Image.NEAREST) 
#     img = img.convert("RGB")  # Convert to RGB for colored text

#     draw = ImageDraw.Draw(img)
#     time_text = f"Time: {i}"  
#     draw.text((10, 10), time_text, fill=(255, 0, 0)) 

#     frames.append(img)

# frames[0].save('animated_with_time.gif', save_all=True, append_images=frames[1:], duration=60, loop=0)

##############################
##############################