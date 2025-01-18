import numpy as np
from collections.abc import Callable
from typing import Tuple

def velocity_verlet(
    initial_position: float,
    initial_momentum: float,
    force_function: Callable[[float, float], float],
    mass: float,
    omega : float,
    timestep: float,
    total_time: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements the Velocity Verlet algorithm.
    """
    time = np.arange(0, total_time, timestep)
    position = np.zeros(len(time))
    momentum = np.zeros(len(time))
    force = np.zeros(len(time))
    position[0] = initial_position
    momentum[0] = initial_momentum
    force[0] = force_function(omega, position[0])
    for t in range(1, len(time)):
        position[t] = position[t - 1] + ((momentum[t - 1] / mass) * timestep) + force[t - 1] / (2 * mass) * timestep * timestep
        force[t] = force_function(omega, position[t])
        momentum[t] = momentum[t - 1] + (1 / (2 * mass)) * (force[t - 1] + force[t]) * timestep
    return position, momentum    

def gear_5th_order_predictor_corrector(
    timestep : float,
    initial_position : float ,
    initial_momentum : float,
    initial_jerk : float,
    initial_snap : float,
    mass : float,
    omega : float,
    force_function : Callable[[float, float], float],
    total_time : int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements the Gear predictor-corrector algorithm of the 5th order
    """
    gear0 = 19.0 / 120.0
    gear1 = 3.0 / 4.0
   #gear2 = 1
    gear3 = 1.0 / 2.0
    gear4 = 1.0 / 12.0

    c1 = timestep
    c2 = c1 * timestep / 2.0
    c3 = c2 * timestep / 3.0
    c4 = c3 * timestep / 4.0

    time = np.arange(0, total_time, timestep)
    position_array = np.zeros(len(time))
    velocity_array = np.zeros(len(time))
    acceleration_array = np.zeros(len(time))
    jerk_array = np.zeros(len(time))
    snap_array = np.zeros(len(time))

    position_array[0] = initial_position
    velocity_array[0] = initial_momentum / mass
    acceleration_array[0] = force_function(omega, initial_position)
    jerk_array[0] = initial_jerk
    snap_array[0] = initial_snap
    
    for t in range(1, len(time)):
        
        position_array[t] = position_array[t-1] + c1 * velocity_array[t-1] + c2 * acceleration_array[t-1] + c3 * jerk_array[t-1] + c4 * snap_array[t-1]
        velocity_array[t] = velocity_array[t-1] + c1 * acceleration_array[t-1] + c2 * jerk_array[t-1] + c3 * snap_array[t-1]
        acceleration_array[t] = acceleration_array[t-1] + c1 * jerk_array[t-1] + c2 * snap_array[t-1]
        jerk_array[t] = jerk_array[t-1] + c1 * snap_array[t-1]

        force = force_function(omega, position_array[t])
        new_acceleration = force / mass
        correction = new_acceleration - acceleration_array[t]

        position_array[t] += gear0 * correction
        velocity_array[t] += gear1 * correction
        acceleration_array[t] = new_acceleration
        jerk_array[t] += gear3 * correction
        snap_array[t] += gear4 * correction

    return position_array, velocity_array
