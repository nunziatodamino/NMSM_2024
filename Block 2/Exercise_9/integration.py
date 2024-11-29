import numpy as np
from collections.abc import Callable
from typing import Tuple

def velocity_verlet(
    initial_position: float,
    initial_momentum: float,
    force_function: Callable[[float], float],
    mass: float,
    timestep: float,
    total_time: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Implements the Velocity Verlet algorithm.

    Args:
        initial_position: Initial position of the particle.
        initial_momentum: Initial momentum of the particle.
        force_function: Callable that computes force as a function of position.
        mass: Mass of the particle.
        timestep: Time step for integration.
        total_time: Total simulation time.

    Returns:
        A tuple of position and momentum arrays over time.
    """
    time = np.arange(0, total_time, timestep)
    position = np.zeros(len(time))
    momentum = np.zeros(len(time))
    force = np.zeros(len(time))
    position[0] = initial_position
    momentum[0] = initial_momentum
    force[0] = force_function(position[0])
    for t in range(1, len(time)):
        position[t] = position[t - 1] + ((momentum[t - 1] / mass) * timestep) + 0.5 * force[t - 1] / (2 * mass) * timestep * timestep
        force[t] = force_function(position[t])
        momentum[t] = momentum[t - 1] + 0.5 * (1 / (2 * mass)) * (force[t - 1] + force[t]) * timestep
    return position, momentum    
