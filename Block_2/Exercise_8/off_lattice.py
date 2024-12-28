import numpy as np
import matplotlib.pyplot as plt
import numba

# PARAMETERS
# All parameters are in LJ reduced units
EPSILON = 1
SIGMA = 1

def reduced_LJ_length_to_length(length, sigma):
    return length * sigma

def reduced_LJ_density_to_density(density, sigma):
    return density * sigma ** 3    

@numba.njit
def lennard_jones_potential(epsilon: float, sigma: float, distance_squared: float) -> float:
    """
    Computes the Lennard-Jones potential for a given squared distance.
    """
    inv_r2 = sigma ** 2 / distance_squared
    inv_r6 = inv_r2 ** 3
    inv_r12 = inv_r6 ** 2
    return 4 * epsilon * (inv_r12 - inv_r6)

@numba.njit
def energy_tail_correction(number_density : float, sigma : float, sigma_cut : float) -> float:
    r = sigma / sigma_cut
    r_3 = r ** 3
    r_9 = r_3 ** 3
    return (8 / 3) * np.pi * number_density * ( (1 / 3) * r_9 - r_3 )

@numba.njit
def pressure_tail_correction(number_density : float, sigma : float, sigma_cut : float) -> float:
    r = sigma / sigma_cut
    r_3 = r ** 3
    r_9 = r_3 ** 3    
    return (16 / 3) * np.pi * number_density**2 * ( (2 / 3) * r_9 - r_3 )    

@numba.njit
def position_PBC(position : float, box_length : float) -> float:
    return position - np.floor(position / box_length) * box_length

@numba.njit
def distance_PBC(distance : float, box_length : float) -> float:
    return distance - np.round(distance / box_length) * box_length

@numba.njit
def energy_particle(number_density : float, n_particles : int, box_size : float, beta : float, \
           epsilon : float, sigma : float, sigma_cut : float, position : np.ndarray, i_random : int) -> float:
    '''
    Evaluates the energy of a single particle in a LJ (Lennard - Jones) system
    '''
    #def kinetic_energy_particle(beta : float) -> float:
    #    return (3 / 2) / beta

    def potential_energy_particle(position : np.ndarray, i_random : int, box_size : float, n_particles : int, \
                         number_density : float, sigma : float, sigma_cut : float, epsilon : float) -> float :
        total_potential_energy = 0.0
        sigma_cut_squared = sigma_cut ** 2
        for j in range(n_particles):
            if j != i_random:
                delta = position[i_random] - position[j]
                delta -= box_size * np.round(delta / box_size)
                distance_squared = np.sum(delta ** 2, dtype = np.float64)
                if distance_squared < sigma_cut_squared:
                    total_potential_energy += lennard_jones_potential(epsilon, sigma, distance_squared)
        total_potential_energy += energy_tail_correction(number_density, sigma, sigma_cut)
        return total_potential_energy

    #kin_energy = kinetic_energy_particle(beta)
    pot_energy = potential_energy_particle(position, i_random, box_size, n_particles, number_density, sigma, sigma_cut, epsilon)
    return pot_energy

@numba.njit
def energy(number_density : float, n_particles : int, box_size : float, beta : float, \
           epsilon : float, sigma : float, sigma_cut : float, position : np.ndarray) -> float:
    '''
    Evaluates the energy of a LJ (Lennard - Jones) system
    '''
    def kinetic_energy(beta : float , n_particles : int) -> float:
        return (3 / 2) * n_particles / beta

    def potential_energy(position : np.ndarray, box_size : float, n_particles : int, \
                         number_density : float, sigma : float, sigma_cut : float, epsilon : float) -> float :
        total_potential_energy = 0.0
        sigma_cut_squared = sigma_cut ** 2
        for i in range(n_particles):
            for j in range(i+1, n_particles):
                delta = position[i] - position[j]
                delta -= box_size * np.round(delta / box_size)
                distance_squared = np.sum(delta ** 2, dtype = np.float64)
                if distance_squared < sigma_cut_squared:
                    total_potential_energy += lennard_jones_potential(epsilon, sigma, distance_squared)
        total_potential_energy += n_particles * energy_tail_correction(number_density, sigma, sigma_cut)
        return total_potential_energy

    kin_energy = kinetic_energy(beta, n_particles)
    pot_energy = potential_energy(position, box_size, n_particles, number_density, sigma, sigma_cut, epsilon)
    return kin_energy + pot_energy

@numba.njit
def metropolis (old_configuration : np.ndarray, new_configuration : np.ndarray, i_random : int, \
                beta : float, n_particles : int, number_density : float, box_size : float, sigma_cut : float) -> bool:
    '''
    Metropolis filter
    '''
    delta_energy = energy_particle(number_density, n_particles, box_size, beta, EPSILON, SIGMA, sigma_cut, new_configuration, i_random) - \
                   energy_particle(number_density, n_particles, box_size, beta, EPSILON, SIGMA, sigma_cut, old_configuration, i_random)
    if delta_energy <= 0: return True
    elif np.random.random() > np.exp(- beta * delta_energy ) : return False
    return True 

@numba.njit
def local_move(n_particles : int, number_density : float , box_size : float, \
                sigma_cut : float, beta: float , position : np.ndarray) -> tuple[bool, np.ndarray] :
    """
    Performs one local move : one particle is chosen at random and a uniform displacement between -d_max and d_max is performed.
    """
    max_displacement = box_size / 100
    old_position = position.copy()
    new_position = position.copy()
    i_random = np.random.randint(0, n_particles)
    new_position[i_random][0] += np.random.uniform(-max_displacement, max_displacement)
    new_position[i_random][1] += np.random.uniform(-max_displacement, max_displacement)
    new_position[i_random][2] += np.random.uniform(-max_displacement, max_displacement)
    if metropolis(old_position, new_position, i_random, beta, n_particles, number_density, box_size, sigma_cut) :
        new_position[i_random][0] = position_PBC(new_position[i_random][0], box_size)
        new_position[i_random][1] = position_PBC(new_position[i_random][1], box_size)
        new_position[i_random][2] = position_PBC(new_position[i_random][2], box_size)
        return True, new_position
    return False, old_position

@numba.njit
def displacement_function(n_particles : int, number_density : float , box_size : float , \
                          sigma_cut : float , beta: float , position : np.ndarray) -> tuple[int, np.ndarray] :
    """
    Performs one Monte Carlo step : one particle is chosen at random and a uniform displacement between -d_max and d_max is performed.
    This is repeated N times when N is the particle number.
    """
    counter = 0
    new_position = position.copy()
    for _ in range(n_particles):
        check, new_position = local_move(n_particles, number_density, box_size, sigma_cut, beta, new_position)
        if check : counter += 1      
    return counter, new_position

@numba.njit
def pressure(number_density : float, box_size : float, temperature : float, \
             epsilon : float, sigma : float, sigma_cut : float, position : np.ndarray) -> float:
    
    def virial(epsilon : float, sigma : float, distance_squared : float) -> float:
        eps_48 = 48 * epsilon 
        a = (sigma ** 2) / distance_squared
        a_6 = a ** 3
        a_12 = a_6 ** 2
        return eps_48 * (a_12 - 0.5 * a_6)

    total_virial = 0
    sigma_cut_squared = sigma_cut**2
    N = len(position[0])
    for i in range(N):
        for j in range(i + 1, N):
            delta = position[i] - position[j]
            delta -= box_size * np.round(delta / box_size)
            distance_squared = np.sum(delta ** 2, dtype = np.float64)
            if distance_squared < sigma_cut_squared:
                total_virial += virial(epsilon, sigma, distance_squared)
    ideal_gas_term = number_density * temperature 
    virial_term = total_virial / (3 * (box_size ** 3))
    correction_term = N * pressure_tail_correction(number_density, sigma, sigma_cut)
    print(ideal_gas_term)
    print(virial_term)
    print(correction_term)
    total_pressure =  ideal_gas_term + virial_term + correction_term
    return total_pressure
    
def plot_particles(position, box_size):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(position[:, 0], position[:, 1], position[:, 2], c='blue', marker='o', label='Particles')
    ax.set_xlim([0, box_size])
    ax.set_ylim([0, box_size])
    ax.set_zlim([0, box_size])

    r = [0, box_size]
    for s, e in zip(
        [[0, 0, 0], [0, 0, box_size], [0, box_size, 0], [0, box_size, box_size], [box_size, 0, 0], [box_size, 0, box_size], [box_size, box_size, 0], [box_size, box_size, box_size]],
        [[0, 0, box_size], [0, box_size, box_size], [box_size, 0, box_size], [box_size, box_size, box_size], [box_size, 0, box_size], [box_size, box_size, box_size], [box_size, box_size, 0], [0, box_size, box_size]]
    ):
        ax.plot3D(*zip(s, e), color="black")

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Particles Inside a Cubic Box')
    plt.legend()
    plt.show()

