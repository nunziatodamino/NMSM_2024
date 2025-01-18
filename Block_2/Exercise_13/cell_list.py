import numpy as np
import matplotlib.pyplot as plt
import numba

K_BOLTZMANN = 1
TEMPERATURE = 1

def particles_initialization(
    particles_number : int,
    radius_1 : float,
    radius_2 : float,
    square_size : float
)-> np.ndarray:
    
    initial_position = np.zeros((particles_number, 2))
    radii_list = np.zeros(particles_number)

    for n in range(particles_number):
        if n < (particles_number // 2): radii_list[n] = radius_1
        else: radii_list[n] = radius_2

    for n in range(particles_number):
        initialized = False
        while (not initialized):
            x_trial = np.random.uniform(radii_list[n], square_size - radii_list[n])
            y_trial = np.random.uniform(radii_list[n], square_size - radii_list[n])
            if n == 0 :
                initial_position[n][0] = x_trial
                initial_position[n][1] = y_trial
                initialized = True
            else:
                position_trial = np.array([x_trial, y_trial])
                distances = np.linalg.norm(initial_position[:n] - position_trial, axis=1 )
                if np.all(distances >= (radii_list[n] + radii_list[:n])):
                        initial_position[n] = position_trial
                        initialized = True
    
    return initial_position, radii_list                    

#@numba.njit
def cell_linked_list(
    square_size : float,
    cell_length : float,
    position_list : np.ndarray    
    ):

    if square_size % cell_length != 0:
        raise ValueError(f"Invalid dimensions: square_size ({square_size}) is not divisible by cell_length ({cell_length}).")
    
    num_cells_per_dim = int(square_size / cell_length)
    cell_number = num_cells_per_dim ** 2

    head_list = np.full(cell_number, -1, dtype=np.int64)  
    cell_list = np.full(len(position_list), -1, dtype=np.int64)              
    
    for particle_num, position in enumerate(position_list):
        cell_x = int(np.floor(position[0] / cell_length))
        cell_y = int(np.floor(position[1] / cell_length))
        c = cell_x + num_cells_per_dim * cell_y   
        cell_list[particle_num] = head_list[c]
        head_list[c] = particle_num

    return cell_list, head_list

#@numba.njit
def neighbor_list_square(square_size: float, cell_length: float):
    """Create a neighbor map for all cells without applying periodic boundary conditions (PBC)."""
    
    if square_size % cell_length != 0:
        raise ValueError(
            f"Invalid dimensions: square_size ({square_size}) is not divisible by cell_length ({cell_length})."
        )
    
    num_cells_per_dim = int(square_size / cell_length)
    neighbors_list = []
    
    for cell_x in range(num_cells_per_dim):
        for cell_y in range(num_cells_per_dim):
            neighbors = []
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue 
                    nx = cell_x + dx
                    ny = cell_y + dy
                    if 0 <= nx < num_cells_per_dim and 0 <= ny < num_cells_per_dim:
                        neighbors.append(nx + (ny * num_cells_per_dim))
            neighbors_list.append(neighbors)
    
    return neighbors_list

#@numba.njit
def repulsive_potential(force_constant, distance_squared, ref_distance):
    return force_constant * np.exp (- distance_squared / (2 * ref_distance**2) )

def force_evaluation(
    square_size : float,
    cell_length : float,
    particle_number,
    force_constant,
    position_list : np.ndarray,
    radii_list,
    neighbor_list,
    cell_list : np.ndarray ,
    head_list : np.ndarray
    ):

    if square_size % cell_length != 0:
        raise ValueError(f"Invalid dimensions: square_size ({square_size}) is not divisible by cell_length ({cell_length}).")
    
    num_cells_per_dim = int(square_size / cell_length)
    cell_number = num_cells_per_dim ** 2

    force_x_list = np.zeros(particle_number)
    force_y_list = np.zeros(particle_number)

    for c in range(cell_number):
        for neighbor in neighbor_list[c]:
            i = head_list[c]
            while i != -1:
                j = head_list[neighbor]
                while j != -1:
                    if i < j:  
                        dx = position_list[i][0] - position_list[j][0]
                        dy = position_list[i][1] - position_list[j][1]
                        distance_sq = dx**2 + dy**2
                        ref_distance = radii_list[i] + radii_list[j]
                        if distance_sq < (16 * ref_distance**2):
                            force_x_list[i] +=  (dx / ref_distance**2) * repulsive_potential(force_constant, distance_sq, ref_distance)
                            force_y_list[i] +=  (dy / ref_distance**2) * repulsive_potential(force_constant, distance_sq, ref_distance)  
                    j = cell_list[j]
                i = cell_list[i]
                
    return force_x_list, force_y_list

#@numba.njit
def first_order_integrator_component(
    initial_position_component : float, 
    force_time_list_component : float,
    mobility_particle : float,
    square_size : float, 
    step_number : int, 
    final_time : float
    ):

    time_list = np.linspace(0, final_time, step_number)
    time_step = time_list[1] - time_list[0]
    time_step_root = np.sqrt(time_step)
    
    random_number = np.random.normal(loc = 0, scale = time_step_root )
    factor_1 = mobility_particle * force_time_list_component
    factor_2 = np.sqrt(2 * K_BOLTZMANN * TEMPERATURE * mobility_particle)
    position_component = initial_position_component + time_step * factor_1 + random_number * factor_2
    radius = 1 / mobility_particle
    if position_component < radius:  # Reflecting off the left boundary
        position_component = radius
    elif position_component > (square_size - radius):  # Reflecting off the right boundary
        position_component = square_size - radius
    return position_component 

def position_check(
    square_size : float,
    cell_length : float,
    position_list : np.ndarray,
    radii_list,
    neighbor_list,
    cell_list : np.ndarray ,
    head_list : np.ndarray
    ):

    if square_size % cell_length != 0:
        raise ValueError(f"Invalid dimensions: square_size ({square_size}) is not divisible by cell_length ({cell_length}).")
    
    num_cells_per_dim = int(square_size / cell_length)
    cell_number = num_cells_per_dim ** 2

    index_list = []

    for c in range(cell_number):
        for neighbor in neighbor_list[c]:
            i = head_list[c]
            while i != -1:
                j = head_list[neighbor]
                while j != -1:
                    if i < j:  
                        dx = position_list[i][0] - position_list[j][0]
                        dy = position_list[i][1] - position_list[j][1]
                        distance = np.sqrt(dx**2 + dy**2)
                        ref_distance = radii_list[i] + radii_list[j]
                        if distance < ref_distance : 
                            if i not in index_list : index_list.append(i) # Due to overlapping i<j check is not sufficient
                    j = cell_list[j]
                i = cell_list[i]
    if index_list == [] : return [], False
    return index_list, True

def plt_particles(position, radii_list, particle_number, square_size):
    plt.figure(figsize=(8, 8))
    plt.xlim(0, square_size)
    plt.ylim(0, square_size)
    plt.gca().set_aspect('equal', adjustable='box')

    # Draw particles
    for n in range(particle_number):
        circle = plt.Circle((position[n][0], position[n][1]), radii_list[n], edgecolor='black', facecolor='blue', alpha=0.5)
        plt.gca().add_patch(circle)

    plt.title(f"Particle Positions with Radii (N={particle_number})")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.grid(True)
    plt.show()

@numba.njit
def check_overlap(positions, radii_list):
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = np.sqrt(dx**2 + dy**2)
            radii_sum = radii_list[i] + radii_list[j]
            if dist < radii_sum:
                return i, True
    return -1, False    