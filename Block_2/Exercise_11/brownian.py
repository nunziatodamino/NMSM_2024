import numpy as np
import numba

MASS = 1
K_BOLTZMANN = 1

MASS_UNIT = 1    
LENGTH_UNIT = 1
ENERGY_UNIT = 1

@numba.njit
def first_order_integrator_component(
    initial_position_component : float, 
    force_constant, force_component_function , 
    temperature : float, 
    friction_coeff : float, 
    box_size : float, 
    step_number : int, 
    final_time : float
    ):
    """
    Implements the Euler-Maruyama algorithm for one position component
    """
    position_time_list_component = np.zeros(step_number)
    position_time_list_component_unwrapped = np.zeros(step_number)
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
    position_time_list_component_unwrapped = position_time_list_component.copy()
    position_time_list_component -= np.floor(position_time_list_component / box_size) * box_size
    return position_time_list_component , position_time_list_component_unwrapped   

@numba.njit
def force_harmonic_trap_component(harmonic_constant : float, position_list : np.ndarray):
    return - (harmonic_constant * position_list)  

@numba.njit    
def brownian_motion(
        initial_position : np.ndarray,
        force_constant, 
        force_component_function,
        temperature : float, 
        friction_coeff : float, 
        box_size : float, 
        step_number : int, 
        final_time : float,
        particle_number : int
        ):
    """
    Routine for the brownian motion evolution. Evaluates the mean square displacemt
    """
    msd = np.zeros(step_number)
    for _ in range(particle_number):
        _, position_x_un = first_order_integrator_component(initial_position[0], force_constant, force_component_function, temperature, friction_coeff, box_size, step_number, final_time)
        _, position_y_un = first_order_integrator_component(initial_position[1], force_constant, force_component_function, temperature, friction_coeff, box_size, step_number, final_time)
        _, position_z_un = first_order_integrator_component(initial_position[2], force_constant, force_component_function, temperature, friction_coeff, box_size, step_number, final_time)

        for t in range(step_number):
            msd[t] += np.mean(position_x_un[:t+1]**2 + position_y_un[:t+1]**2 + position_z_un[:t+1]**2)

    msd_total = msd / particle_number

    return msd_total

    
