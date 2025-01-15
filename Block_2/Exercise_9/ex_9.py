import numpy as np
import matplotlib.pyplot as plt
import integration as md

def harmonic_force(x: float) -> float:
    omega = 0.1
    return -omega**2 * x 

# Simulation parameters
initial_position = 1.0
initial_momentum = 0.0
initial_jerk = 0
initial_snap = 0
mass = 1.0
timestep = 0.001
total_time = 100

position, momentum = md.velocity_verlet(initial_position, initial_momentum, harmonic_force, mass, timestep, total_time)
position2, velocity2 = md.gear_5th_order_predictor_corrector(timestep, initial_position, initial_momentum, initial_jerk, initial_snap, mass, harmonic_force, total_time)


time = np.arange(0, total_time, timestep)

plt.figure(figsize=(10, 6))
plt.plot(time, position, label='Position')
plt.plot(time, momentum / mass, label='Velocity (Momentum/Mass)')
plt.plot(time, position2, label='Position PC')
plt.plot(time, velocity2, label='Velocity PC (Momentum/Mass)')
plt.xlabel('Time')
plt.ylabel('Position / Velocity')
plt.title('Velocity Verlet Simulation')
plt.legend()
plt.grid()
plt.show()