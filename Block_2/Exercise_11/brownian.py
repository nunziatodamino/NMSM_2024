import numpy as np
import numba

MASS = 1
K_BOLTZMANN = 1

MASS_UNIT = 1    
LENGTH_UNIT = 1
ENERGY_UNIT = 1

def first_order_integrator_component(initial_position_component : float, force_constant, force_component_function , temperature : float, friction_coeff : float, \
                           box_size : float, step_number : int, final_time : float):
    position_time_list_component = np.zeros(step_number)
    position_time_list_component[0] = initial_position_component
    time_list = np.linspace(0, final_time, step_number)
    time_step = time_list[1] - time_list[0]
    time_step_root = np.sqrt(time_step)
    for t in range(1, len(time_list)):
        random_number = np.random.normal(loc = 0, scale = time_step_root )
        force_time_list_component = force_component_function(force_constant, position_time_list_component[t-1])
        factor_1 = force_time_list_component / (MASS * friction_coeff)
        factor_2 = np.sqrt((2 * K_BOLTZMANN * temperature)/(MASS * friction_coeff))
        position_time_list_component[t] = position_time_list_component[t-1] + time_step * factor_1 + random_number * factor_2
        position_time_list_component[t] -= np.floor(position_time_list_component[t] / box_size) * box_size
    return position_time_list_component    

def force_harmonic_trap_component(harmonic_constant : float, position_list : np.ndarray):
    return - (harmonic_constant * position_list)  
    

