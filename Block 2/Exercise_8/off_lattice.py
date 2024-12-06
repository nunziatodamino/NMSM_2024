import numpy as np
import matplotlib.pyplot as plt
import numba

BOX_SIZE = 10
SIGMA_CUT = BOX_SIZE / 2
EPSILON = 1
SIGMA = 1
PARTICLE_NUMBER = 10
NUMBER_DENSITY = PARTICLE_NUMBER / BOX_SIZE**2


def lennard_jones_potential(epsilon : float, sigma : float, sigma_cut : float, distance : float) -> float: 
    if distance >= sigma_cut:
        return 0
    return 4 * epsilon * ( (sigma / distance)**12 - (sigma / distance)**6 )

def energy_tail_correction(number_density : float, sigma : float, sigma_cut : float) -> float:
    return (8 / 3) * np.pi * number_density * ( (1 / 3) * (sigma / sigma_cut)**9 -(sigma / sigma_cut)**3 )

def pressure_tail_correction(number_density : float, sigma : float, sigma_cut : float) -> float:
    return (16 / 3) * np.pi * number_density**2 * ( (2 / 3) * (sigma / sigma_cut)**9 -(sigma / sigma_cut)**3 )    

def position_PBC(position : float, box_length : float) -> float:
    return position - np.floor(position / box_length) * box_length

def energy(number_density : float, n_particles : float, beta : float,  epsilon : float, sigma : float, sigma_cut : float, position : np.ndarray):
    '''
    Evaluates the energy of a LJ (Lennard - Jones) system
    '''
    def kinetic_energy(beta, n_particles):
        return 3 / 2 * n_particles / beta
    def potential_energy(position):
        total_potential_energy = 0
        for i in range(len(position[0])):
            for j in range(len(position[0])):
                if i !=j :
                    delta_x = position[i][0] - position[j][0]
                    delta_y = position[i][1] - position[j][1]
                    delta_z = position[i][2] - position[j][2]
                    distance = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
                    if distance < sigma_cut:
                        total_potential_energy += lennard_jones_potential(epsilon, sigma, sigma_cut, distance)
                    else : total_potential_energy += 2 * energy_tail_correction(number_density, sigma, sigma_cut)
        return total_potential_energy
    return kinetic_energy(beta, n_particles) + potential_energy(position)

def metropolis (old_configuration : np.ndarray, new_configuration : np.ndarray, beta : float) -> bool:
    '''
    Metropolis filter
    '''
    delta_Energy=energy(NUMBER_DENSITY, PARTICLE_NUMBER, beta, EPSILON, SIGMA, SIGMA_CUT, new_configuration)-energy(NUMBER_DENSITY, PARTICLE_NUMBER, beta, EPSILON, SIGMA, SIGMA_CUT, old_configuration)
    if delta_Energy <= 0: return True
    elif np.random.random() > np.exp(- beta * delta_Energy ) : return False
    return True 

def displacement_function(n_particles : int, box_size : float, beta: float , position : np.ndarray) -> np.ndarray :
    max_displacement = box_size / 10
    old_position = position.copy()
    new_position = position.copy()
    for i in range(n_particles):
        new_position[i][0] += np.random.uniform(-max_displacement, max_displacement)
        new_position[i][0] = position_PBC(new_position[i][0], box_size)
        new_position[i][1] += np.random.uniform(-max_displacement, max_displacement)
        new_position[i][1] = position_PBC(new_position[i][1], box_size)
        new_position[i][2] += np.random.uniform(-max_displacement, max_displacement)
        new_position[i][2] = position_PBC(new_position[i][2], box_size)
    if metropolis(old_position, new_position, beta) : return new_position
    return old_position

def plot_particles(position, BOX_SIZE):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(position[:, 0], position[:, 1], position[:, 2], c='blue', marker='o', label='Particles')
    ax.set_xlim([0, BOX_SIZE])
    ax.set_ylim([0, BOX_SIZE])
    ax.set_zlim([0, BOX_SIZE])

    r = [0, BOX_SIZE]
    for s, e in zip(
        [[0, 0, 0], [0, 0, BOX_SIZE], [0, BOX_SIZE, 0], [0, BOX_SIZE, BOX_SIZE], [BOX_SIZE, 0, 0], [BOX_SIZE, 0, BOX_SIZE], [BOX_SIZE, BOX_SIZE, 0], [BOX_SIZE, BOX_SIZE, BOX_SIZE]],
        [[0, 0, BOX_SIZE], [0, BOX_SIZE, BOX_SIZE], [BOX_SIZE, 0, BOX_SIZE], [BOX_SIZE, BOX_SIZE, BOX_SIZE], [BOX_SIZE, 0, BOX_SIZE], [BOX_SIZE, BOX_SIZE, BOX_SIZE], [BOX_SIZE, BOX_SIZE, 0], [0, BOX_SIZE, BOX_SIZE]]
    ):
        ax.plot3D(*zip(s, e), color="black")

    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_zlabel('Z-axis')
    ax.set_title('Particles Inside a Cubic Box')
    plt.legend()
    plt.show()