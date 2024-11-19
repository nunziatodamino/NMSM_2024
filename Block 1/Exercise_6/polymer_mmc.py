import numpy as np
import numba

# Parameters
c = 1 
monomer_radius = c / 2
x_initial,y_initial = (0, 0)
bond_length_min, bond_length_max = (1, 1.3)
n_monomers = 20
epsilon_energy = 1
energy_threshold = 0.5

@numba.njit
def polymer_displacement(configuration):
    sigma = monomer_radius / 8
    shift_conf = np.zeros((n_monomers,2))
    valid_configuration = False
    # ---- Move generation
    while (not valid_configuration): # Local shifts move
        for i in range(1,n_monomers):
            shift_conf[i][0]=configuration[i][0] + np.random.normal(0, sigma)
            shift_conf[i][1]=configuration[i][1] + np.random.normal(0, sigma)
        MAX_ANGLE = np.pi / 6 # Pivot rotation move
        angle = np.random.uniform(-MAX_ANGLE, MAX_ANGLE)
        pivot = np.random.randint(0, n_monomers)  # Ensure pivot is an integer index
        new_conf = shift_conf   
        for i in range(pivot, n_monomers): # Apply rotation around the pivot point
            dx = shift_conf[i][0] - shift_conf[pivot][0]
            dy = shift_conf[i][1] - shift_conf[pivot][1]
            new_conf[i] = (
                dx * np.cos(angle) - dy * np.sin(angle) + shift_conf[pivot][0],
                dx * np.sin(angle) + dy * np.cos(angle) + shift_conf[pivot][1]
            )
    # ----- Geometry constraints
        valid_configuration = True # Check bond length constraint
        for i in range(n_monomers - 1):
            bond_length = np.sqrt((new_conf[i+1][0] - new_conf[i][0]) ** 2 + (new_conf[i+1][1] - new_conf[i][1]) ** 2)
            if not (bond_length_min <= bond_length <= bond_length_max):
                valid_configuration = False
        for monomer in new_conf: # Check hard wall constraint
            if monomer[1] < 0: 
                valid_configuration = False
        for i in range(n_monomers):# Check minimum distance between monomer centers
            for j in range(i + 1, n_monomers):
                center_distance = np.sqrt((new_conf[i][0] - new_conf[j][0]) ** 2 + (new_conf[i][1] - new_conf[j][1]) ** 2)
                if center_distance < 2 * monomer_radius:
                        valid_configuration = False
    return new_conf

@numba.njit(cache=True)
def end2end_distance_squared (configuration : np.ndarray) -> np.float64:
    return np.sqrt((configuration[-1][0]- configuration[0][0])**2+(configuration[-1][1]- configuration[0][1])**2)

@numba.njit(cache=True)
def energy (configuration : np.ndarray) -> np.float64:
    energy = 0
    for point in configuration:
        if point[1]<energy_threshold: energy-=epsilon_energy
    return energy

@numba.njit(cache=True)
def gyration_radius (configuration : np.ndarray) -> np.float64:
    total_sum = 0 
    for point in configuration:
        total_sum += point[0]**2 + point[1]**2
    return np.sqrt(total_sum / n_monomers)

@numba.njit(cache=True)
def metropolis (old_configuration : np.ndarray, new_configuration : np.ndarray, beta : float) -> bool:
    delta_Energy=energy(new_configuration)-energy(old_configuration)
    if delta_Energy <= 0: return True
    elif np.random.random() <= np.exp(- beta * delta_Energy ) : return True
    return False            

@numba.njit
def thermalization(monomers_initial_conf: np.ndarray, time_max : np.int32, beta : np.float64) -> (np.ndarray,np.ndarray,np.ndarray) :
    moves=np.zeros((time_max, n_monomers, 2))
    moves[0]=monomers_initial_conf
    ee2=np.zeros((time_max))
    energy_list=np.zeros((time_max))
    end_heigth=np.zeros((time_max))
    gyr_radius_list = np.zeros((time_max))
    for i in range(1, time_max):
        moves[i] = polymer_displacement(moves[i-1])
        if not metropolis(moves[i-1],moves[i], beta): moves[i] = moves[i-1]
        ee2[i] = end2end_distance_squared(moves[i])
        end_heigth[i]=moves[i,n_monomers-1,1]
        energy_list[i]=energy(moves[i])
        gyr_radius_list[i] = gyration_radius(moves[i])
    return ee2, end_heigth, energy_list, gyr_radius_list
