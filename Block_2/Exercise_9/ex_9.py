import numpy as np
import matplotlib.pyplot as plt
import integration as md

def harmonic_force(x: float) -> float:
    omega = 0.1
    return -omega**2 * x 

# Simulation parameters
initial_position = 1.0
initial_momentum = 0.0
mass = 1.0
timestep = 0.01
total_time = 100.0

position, momentum = md.velocity_verlet(initial_position, initial_momentum, harmonic_force, mass, timestep, total_time)

time = np.arange(0, total_time, timestep)

plt.figure(figsize=(10, 6))
plt.plot(time, position, label='Position')
plt.plot(time, momentum / mass, label='Velocity (Momentum/Mass)')
plt.xlabel('Time')
plt.ylabel('Position / Velocity')
plt.title('Velocity Verlet Simulation')
plt.legend()
plt.grid()
plt.show()