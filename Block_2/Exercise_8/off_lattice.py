import numpy as np
import numba

# REFERENCE UNITS
EPSILON = 1
SIGMA = 1

def reduced_LJ_length_to_length(length, sigma):
    return length * sigma

def reduced_LJ_density_to_density(density, sigma):
    return density * sigma ** 3    

@numba.njit
def lennard_jones_potential(distance_squared: float) -> float:
    """
    Computes the Lennard-Jones potential for a given squared distance.
    """
    inv_r2 = (SIGMA ** 2) / distance_squared
    prefactor = 4 * EPSILON 
    return prefactor * (inv_r2 ** 6 - inv_r2 ** 3)

@numba.njit
def energy_tail_correction(
    number_density : float, 
    sigma_cut : float
    )-> float:
    """
    Energy tail correction to be applied per particle
    """
    r = SIGMA / sigma_cut
    r_3 = r ** 3
    r_9 = r_3 ** 3
    return (8 / 3) * np.pi * number_density * ( ((1 / 3) * r_9) - r_3 )

@numba.njit
def pressure_tail_correction(
    number_density : float, 
    sigma_cut : float
    ) -> float:
    """
    Pressure tail correction to be applied once for the system
    """
    r = SIGMA / sigma_cut
    prefactor = (16 / 3) * np.pi * (number_density ** 2)     
    return  prefactor * ( ((2 / 3) * (r ** 9)) - (r ** 3) )    

@numba.njit
def position_PBC(position : float, box_length : float) -> float:
    return position - np.floor(position / box_length) * box_length

@numba.njit
def distance_PBC(distance : float, box_length : float) -> float:
    return distance - np.round(distance / box_length) * box_length

@numba.njit
def energy_particle(
    n_particles : int, 
    box_size : float, 
    sigma_cut : float, 
    position : np.ndarray, 
    i_random : int
    ) -> float:
    '''
    Evaluates the potential energy of a single particle in a LJ (Lennard - Jones) system
    '''
    total_potential_energy = 0.0
    sigma_cut_squared = sigma_cut ** 2
    for j in range(n_particles):
        if j != i_random:
            delta_x = position[i_random][0] - position[j][0]
            delta_x -= box_size * np.round(delta_x / box_size)
            delta_y = position[i_random][1] - position[j][1]
            delta_y -= box_size * np.round(delta_y / box_size)    
            delta_z = position[i_random][2] - position[j][2]
            delta_z -= box_size * np.round(delta_z / box_size)
            distance_squared = delta_x**2 +delta_y**2 +delta_z**2
            if distance_squared < sigma_cut_squared:
                total_potential_energy += lennard_jones_potential(distance_squared)
    #total_potential_energy += energy_tail_correction(number_density, sigma, sigma_cut)
    return total_potential_energy

@numba.njit
def metropolis (delta_energy : float, beta : float) -> bool:
    '''
    Metropolis filter
    '''
    if delta_energy < 0: return True
    elif np.random.random() > np.exp(- beta * delta_energy ) : return False
    return True 

@numba.njit
def local_move(
    n_particles : int,
    box_size : float,
    sigma_cut : float, 
    beta: float , 
    position : np.ndarray
    ) -> tuple[bool, np.ndarray] :
    """
    Performs one local move : one particle is chosen at random and a uniform displacement between -d_max and d_max is performed.
    """
    max_displacement = box_size / 50
    old_position = position.copy()
    i_random = np.random.randint(0, n_particles)
    en_old = energy_particle(n_particles, box_size, sigma_cut, old_position, i_random)
    position[i_random][0] += np.random.uniform(-max_displacement, max_displacement)
    position[i_random][1] += np.random.uniform(-max_displacement, max_displacement)
    position[i_random][2] += np.random.uniform(-max_displacement, max_displacement)
    en_new = energy_particle(n_particles, box_size, sigma_cut, position, i_random)
    delta_energy = en_new - en_old
    if metropolis(delta_energy, beta) : 
        position[i_random][0] -= box_size * np.floor(position[i_random][0] / box_size)
        position[i_random][1] -= box_size * np.floor(position[i_random][1] / box_size)
        position[i_random][2] -= box_size * np.floor(position[i_random][2] / box_size)
        return True, position
    return False, old_position

@numba.njit
def displacement_function(
    n_particles : int, 
    box_size : float ,
    sigma_cut : float , 
    beta: float , 
    position : np.ndarray
    ) -> tuple[int, np.ndarray] :
    """
    Performs one Monte Carlo step : one particle is chosen at random and a uniform displacement between -d_max and d_max is performed.
    This is repeated N times when N is the particle number.
    """
    counter = 0
    new_position = position.copy()
    for _ in range(n_particles):
        check, new_position = local_move(n_particles, box_size, sigma_cut, beta, new_position)
        if check : counter += 1      
    return counter, new_position

@numba.njit
def virial(
    box_size: float, 
    sigma_cut: float, 
    position: np.ndarray
    ) -> float:
    """
    Evaluates the virial
    """
    eps_48 = 48 * EPSILON
    sigma_squared = SIGMA ** 2
    sigma_cut_squared = sigma_cut ** 2
    total_virial = 0.0
    N = len(position)

    for i in range(N):
        for j in range(i + 1, N):
            delta_x = position[i][0] - position[j][0]
            delta_x -= box_size * np.rint(delta_x / box_size)
            delta_y = position[i][1] - position[j][1]
            delta_y -= box_size * np.rint(delta_y / box_size)
            delta_z = position[i][2] - position[j][2]
            delta_z -= box_size * np.rint(delta_z / box_size)
            distance_squared = delta_x**2 + delta_y**2 + delta_z**2
            if distance_squared < sigma_cut_squared:
                inv_r2 = sigma_squared / distance_squared
                inv_r6 = inv_r2 ** 3
                inv_r12 = inv_r6 ** 2
                total_virial += inv_r12 - 0.5 * inv_r6
    return eps_48 * total_virial           

@numba.njit
def pressure(
    number_density : float, 
    box_size : float, 
    temperature : float,  
    sigma_cut : float, 
    virial_mean : float
    ) -> float:
    """
    Evaluates the system pressure with a tail correction
    """
    ideal_gas_term = number_density * temperature 
    virial_term = virial_mean / (3 * (box_size ** 3))
    correction_term = pressure_tail_correction(number_density, sigma_cut)
    total_pressure = ideal_gas_term + virial_term + correction_term
    return total_pressure

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


