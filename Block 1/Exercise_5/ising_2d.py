import numpy as np

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

def system_energy(configuration, neighbors_list):
    energy = 0 
    for (i, j), neighbors in neighbors_list.items():
        for neighbor in neighbors:
            neighbor_i, neighbor_j = neighbor
            energy -= J * configuration[i][j] * configuration[neighbor_i][neighbor_j]
    return energy / 2

def metropolis_spin_flip_dynamics (old_configuration, neighbors_list, length, beta):
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

def glauber_spin_flip_dynamics (old_configuration, neighbors_list, length, beta):
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

def normalized_auto_correlation(distribution, t_equilibrium, time_max):
    """
    Autocorrelation function between time t_eq to t_max
    """
    auto_corr = np.zeros(time_max - t_equilibrium)
    for t in range(t_equilibrium, time_max):
        tmp_sum_1 = 0
        tmp_sum_2 = 0
        tmp_sum_3 = 0
        for i in range(time_max - t):
            tmp_sum_1 += distribution[i] * distribution[i+t]
            tmp_sum_2 += distribution[i]
            tmp_sum_3 += distribution[i+t]
        prefactor = 1 / (time_max - t)
        auto_corr[t] = prefactor * tmp_sum_1 - prefactor * tmp_sum_2 * prefactor * tmp_sum_3 
    return auto_corr/auto_corr[t_equilibrium]  

def heat_capacity (energy_variance, temperature):
    return energy_variance / temperature**2

def magnetic_susceptibility( length , beta, magnetisation_per_spin_variance):
    return length * length * beta * magnetisation_per_spin_variance