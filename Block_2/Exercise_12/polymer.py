import numpy as np
import numba

# Parameters
c = 1 
monomer_radius = c / 2
x_initial,y_initial = (0, 0)
bond_length_min, bond_length_max = (1, 1.3)
n_monomers = 35
epsilon_energy = 1
energy_threshold = 0.5

@numba.njit
def polymer_displacement(configuration):
    '''
    Takes as input a configuration (np.ndarray(n_monomers, 2)) and output a configuration of the same type after the following
    operations are performed:
    - N local gaussian shift for each monomer
    - 1 pivot rotation around a uniformly selected monomer
    Then the following checks are done:
    - bond length constraint
    - non compenetration
    - hard wall constrain
    If the function passes all the checks then outputs the configuration, if not restarts
    '''
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
        pivot = np.random.randint(0, n_monomers) 
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
    '''
    Evaluates the end to end distance of the polymer which position coordinates are stored in the configuration array
    '''
    return np.sqrt((configuration[-1][0]- configuration[0][0])**2+(configuration[-1][1]- configuration[0][1])**2)

@numba.njit(cache=True)
def energy (configuration : np.ndarray) -> np.float64:
    '''
    Evaluates the energy of the polymer considering an absorbing wall, which position coordinates are stored in the configuration array
    '''
    energy = 0
    for point in configuration:
        if point[1]<energy_threshold: energy-=epsilon_energy
    return energy

@numba.njit(cache=True)
def gyration_radius (configuration : np.ndarray) -> np.float64:
    '''
    Evaluates the gyration radius of a polymer which position coordinates are stored in the configuration array
    '''
    total_sum = 0 
    for point in configuration:
        total_sum += point[0]**2 + point[1]**2
    return np.sqrt(total_sum / n_monomers)

@numba.njit(cache=True)
def metropolis (old_configuration : np.ndarray, new_configuration : np.ndarray, beta : float) -> bool:
    '''
    Metropolis filter
    '''
    delta_Energy=energy(new_configuration)-energy(old_configuration)
    if delta_Energy <= 0: return True
    elif np.random.random() > np.exp(- beta * delta_Energy ) : return False
    return True            

@numba.njit(cache=True)
def mmc_swap (old_configuration : np.ndarray, new_configuration : np.ndarray, beta_list : np.ndarray , k_selected : np.int32 ) -> bool:
    '''
    Multiple Markov chain (MMC) swap mechanism
    '''
    delta_energy= energy(new_configuration)-energy(old_configuration)
    delta_beta = beta_list[k_selected + 1] - beta_list[k_selected]
    if delta_energy <= 0: return True
    elif np.random.random() > np.exp(- delta_beta * delta_energy ) : return False
    return True         

@numba.njit
def thermalization(monomers_initial_conf: np.ndarray, time_max : np.int32, beta : np.float64) -> (np.ndarray,np.ndarray,np.ndarray) :
    '''
    ATTENTION : MEMORY INTENSIVE
    Makes one move and accept/refuse it with the metropolis filter. 
    Also evaluates the observables energy, end to end, gyration radius and end height
    '''
    moves=np.zeros((time_max, n_monomers, 2))
    moves[0]=monomers_initial_conf
    ee2=np.zeros((time_max))
    energy_list=np.zeros((time_max))
    end_heigth=np.zeros((time_max))
    gyr_radius_list = np.zeros((time_max))
    ee2[0] = end2end_distance_squared(moves[0])
    end_heigth[0]=moves[0,n_monomers-1,1]
    energy_list[0]=energy(moves[0])
    gyr_radius_list[0] = gyration_radius(moves[0])
    for i in range(1, time_max):
        moves[i] = polymer_displacement(moves[i-1])
        if not metropolis(moves[i-1],moves[i], beta): moves[i] = moves[i-1]
        ee2[i] = end2end_distance_squared(moves[i])
        end_heigth[i]=moves[i,n_monomers-1,1]
        energy_list[i]=energy(moves[i])
        gyr_radius_list[i] = gyration_radius(moves[i])
    return ee2, end_heigth, energy_list, gyr_radius_list, moves

@numba.njit
def evolution(monomers_initial_conf: np.ndarray, time_max: np.int32, beta: np.float64):
    '''
    Makes one move and accept/refuse it with the metropolis filter. 
    Also evaluates the observables energy, end to end, gyration radius and end height
    Optimized function for memory.
    '''
    moves = monomers_initial_conf.copy() 
    ee2 = np.zeros(time_max)
    energy_list = np.zeros(time_max)
    end_heigth = np.zeros(time_max)
    gyr_radius_list = np.zeros(time_max)
    for i in range(1, time_max):
        proposed_move = polymer_displacement(moves)
        if metropolis(moves, proposed_move, beta):
            moves[:] = proposed_move
        ee2[i] = end2end_distance_squared(moves)
        end_heigth[i] = moves[-1, 1]  
        energy_list[i] = energy(moves)
        gyr_radius_list[i] = gyration_radius(moves)

    return ee2, end_heigth, energy_list, gyr_radius_list, moves

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

