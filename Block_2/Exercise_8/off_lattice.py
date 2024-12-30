import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
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
    inv_r2 = (sigma ** 2) / distance_squared
    prefactor = 4 * epsilon 
    return prefactor * (inv_r2 ** 6 - inv_r2 ** 3)

@numba.njit
def energy_tail_correction(number_density : float, sigma : float, sigma_cut : float) -> float:
    r = sigma / sigma_cut
    r_3 = r ** 3
    r_9 = r_3 ** 3
    return (8 / 3) * np.pi * number_density * ( ((1 / 3) * r_9) - r_3 )

@numba.njit
def pressure_tail_correction(number_density : float, sigma : float, sigma_cut : float) -> float:
    r = sigma / sigma_cut
    prefactor = (16 / 3) * np.pi * (number_density ** 2)     
    return  prefactor * ( ((2 / 3) * (r ** 9)) - (r ** 3) )    

@numba.njit
def position_PBC(position : float, box_length : float) -> float:
    return position - np.floor(position / box_length) * box_length

@numba.njit
def distance_PBC(distance : float, box_length : float) -> float:
    return distance - np.round(distance / box_length) * box_length

@numba.njit
def energy_particle(n_particles : int, box_size : float, epsilon : float, \
                    sigma : float, sigma_cut : float, position : np.ndarray, i_random : int) -> float:
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
                total_potential_energy += lennard_jones_potential(epsilon, sigma, distance_squared)
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
def local_move(n_particles : int, number_density : float , box_size : float, epsilon : float , sigma : float ,\
                sigma_cut : float, beta: float , position : np.ndarray) -> tuple[bool, np.ndarray] :
    """
    Performs one local move : one particle is chosen at random and a uniform displacement between -d_max and d_max is performed.
    """
    max_displacement = box_size / 50
    old_position = position.copy()
    i_random = np.random.randint(0, n_particles)
    en_old = energy_particle(n_particles, box_size, epsilon, sigma, sigma_cut, old_position, i_random)
    position[i_random][0] += np.random.uniform(-max_displacement, max_displacement)
    position[i_random][1] += np.random.uniform(-max_displacement, max_displacement)
    position[i_random][2] += np.random.uniform(-max_displacement, max_displacement)
    en_new = energy_particle(n_particles, box_size, epsilon, sigma, sigma_cut, position, i_random)
    delta_energy = en_new - en_old
    if metropolis(delta_energy, beta) : 
        position[i_random][0] -= box_size * np.floor(position[i_random][0] / box_size)
        position[i_random][1] -= box_size * np.floor(position[i_random][1] / box_size)
        position[i_random][2] -= box_size * np.floor(position[i_random][2] / box_size)
        return True, position
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
def virial(box_size: float, epsilon: float, sigma: float, sigma_cut: float, position: np.ndarray) -> float:
    eps_48 = 48 * epsilon
    sigma_squared = sigma ** 2
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
def pressure(number_density : float, box_size : float, temperature : float, \
            sigma : float, sigma_cut : float, virial_mean : float) -> float:
    ideal_gas_term = number_density * temperature 
    virial_term = virial_mean / (3 * (box_size ** 3))
    correction_term = pressure_tail_correction(number_density, sigma, sigma_cut)
    print(ideal_gas_term)
    print(virial_term)
    print(correction_term)
    total_pressure = ideal_gas_term + virial_term + correction_term
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

@numba.njit
def energy(number_density : float, n_particles : int, box_size : float, \
           epsilon : float, sigma : float, sigma_cut : float, position : np.ndarray) -> float:
    '''
    Evaluates the energy of a LJ (Lennard - Jones) system
    '''
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


# Function to create a GIF for particle positions in the cubic box
def create_gif_for_positions(position_list, box_size, output_dir="gifs", interval=50, frames=200):
    """
    Creates a GIF showing the evolution of particle positions in the cubic box.

    Parameters:
        position_list (np.ndarray): Array of particle positions, shape (iterations, particle_number, 3).
        box_size (float): Size of the cubic box.
        output_dir (str): Directory to save the GIFs.
        interval (int): Delay between frames in milliseconds.
        frames (int): Number of frames in the GIF.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, box_size])
    ax.set_ylim([0, box_size])
    ax.set_zlim([0, box_size])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    points = ax.scatter([], [], [], s=5, c='blue')

    def update(frame):
        current_positions = position_list[frame % len(position_list)]
        points._offsets3d = (current_positions[:, 0], current_positions[:, 1], current_positions[:, 2])
        ax.set_title(f"Frame {frame}")
        return points,

    anim = FuncAnimation(fig, update, frames=min(frames, len(position_list)), interval=interval, blit=False)
    output_file = os.path.join(output_dir, f"simulation_positions_box_size_{box_size:.2f}.gif")
    anim.save(output_file, writer="imagemagick")
    print(f"Saved GIF: {output_file}")
    plt.close()    

@numba.njit
def displacement_function_naive(n_particles : int, number_density : float , box_size : float , \
                          sigma_cut : float , beta: float , position : np.ndarray) -> tuple[int,np.ndarray] :
    """
    Performs one Monte Carlo step : one particle is chosen at random and a uniform displacement between -d_max and d_max is performed.
    This is repeated N times when N is the particle number.
    """
    max_displacement = box_size / 100
    old_position = position.copy()
    new_position = position.copy()
    counter = 0
    for i in range(n_particles):
        new_position[i][0] += np.random.uniform(-max_displacement, max_displacement)
        new_position[i][1] += np.random.uniform(-max_displacement, max_displacement)
        new_position[i][2] += np.random.uniform(-max_displacement, max_displacement)
        new_position[i][0] -= box_size * np.floor(new_position[i][0] / box_size)
        new_position[i][1] -= box_size * np.floor(new_position[i][1] / box_size)
        new_position[i][2] -= box_size * np.floor(new_position[i][2] / box_size)
        
        if metropolis(old_position, new_position, i, beta, n_particles, number_density, box_size, sigma_cut) :
            counter +=1
            old_position = new_position.copy()
    return counter, new_position
