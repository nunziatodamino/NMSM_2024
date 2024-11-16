import numpy as np
from numba import njit
from statsmodels.tsa.stattools import acf

J = 1

def neighbors_list_square_pbc(length):
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
    neighbors_list = np.zeros((length, length, 4, 2), dtype=np.int32)
    for i in range(length):
        for j in range(length):
            neighbors_list[i, j, 0] = ((i - 1) % length, j)  
            neighbors_list[i, j, 1] = ((i + 1) % length, j)  
            neighbors_list[i, j, 2] = (i, (j - 1) % length)  
            neighbors_list[i, j, 3] = (i, (j + 1) % length)  
    return neighbors_list

def system_energy(configuration, neighbors_list):
    energy = 0 
    for (i, j), neighbors in neighbors_list.items():
        for neighbor in neighbors:
            neighbor_i, neighbor_j = neighbor
            energy -= J * configuration[i][j] * configuration[neighbor_i][neighbor_j]
    return energy / 2

@njit
def system_energy_opt(configuration, neighbors_list, length):
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
    """
    """
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
    i = np.random.randint(0, length)
    j = np.random.randint(0, length)
    neighbor_sum = sum(old_configuration[neighbor] for neighbor in neighbors_list[(i, j)])
    delta_energy = 2 * J * old_configuration[i,j] * neighbor_sum
    new_configuration= old_configuration.copy()
    new_configuration[i,j] = -old_configuration[i,j] # spin flip
    if np.random.random() <= 1 / (1 + np.exp( beta * delta_energy )) : return new_configuration
    return old_configuration

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

def mean_value_observable_equilibrium(observable, t_equilibrium, t_max):
    return 1/(t_max - t_equilibrium) * np.sum(observable[t_equilibrium : t_max])

def variance_observable_equilibrium(observable, t_equilibrium, t_max):
    tmp = 1/(t_max - t_equilibrium) * np.sum(observable[t_equilibrium : t_max]**2)
    return tmp - mean_value_observable_equilibrium(observable, t_equilibrium, t_max)**2

def error_observable_equilibrium(observable, t_equilibrium, t_max):
    return np.sqrt(variance_observable_equilibrium(observable, t_equilibrium, t_max) / (t_max - t_equilibrium))

def heat_capacity(energy_variance, temperature):
    return energy_variance / temperature**2

def magnetic_susceptibility( length , beta, magnetisation_per_spin_variance):
    return length * length * beta * magnetisation_per_spin_variance

def normalized_autocorrelation_function(observable):
    acf = np.correlate(observable, observable, mode = "full")
    return acf/acf[0]

def auto_correlation_wrt(distribution, time_max):    
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