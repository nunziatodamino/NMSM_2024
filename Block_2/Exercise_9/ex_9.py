import os
import numpy as np
import matplotlib.pyplot as plt
import integration as md

######################################################################
image_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block_2/Report/FIG/ex9"
######################################################################

omega = 0.1

def harmonic_force(omega : float, x: float) -> float:    
    return -(omega**2) * x 

def analytical_position (amplitude : float, omega : float, t : float) -> float:
    return amplitude * np.sin(omega * t) 

def analytical_velocity (amplitude : float, omega : float,  t : float) -> float:
    return amplitude * omega *  np.cos(omega * t) 

mass = 1.0
initial_position = 0.0
initial_momentum = 1.0
initial_jerk = 0
initial_snap = 0
timestep = 0.001
total_time = 1000
initial_condition_sq = initial_position**2 + (initial_momentum / mass)**2
amplitude = np.sqrt(initial_condition_sq / omega**2)

position_verlet, momentum_verlet = md.velocity_verlet(initial_position, initial_momentum, harmonic_force, mass, omega, timestep, total_time)
position_gear, velocity_gear = md.gear_5th_order_predictor_corrector(timestep, initial_position, initial_momentum, initial_jerk, initial_snap, mass, omega, harmonic_force, total_time)

time = np.arange(0, total_time, timestep)
an_position = analytical_position(amplitude, omega, time)
an_velocity = analytical_velocity(amplitude, omega, time)

plt.figure(figsize=(18, 6))
plt.plot(time, position_verlet, label='Position Verlet')
plt.plot(time, momentum_verlet / mass, label='Velocity Verlet (Momentum/Mass)')
plt.plot(time, an_position, label='Analytical position')
plt.plot(time, an_velocity, label='Analytical velocity ')
plt.plot(time, position_gear, label='Gear 5th position PC')
plt.plot(time, velocity_gear, label='Gear 5th velocity PC')
plt.xlabel('Time')
plt.ylabel('Position / Velocity')
plt.legend()
image_name = f"integration_schemes_comparison"
full_path = os.path.join(image_path, image_name)
plt.savefig(full_path)
plt.show()
plt.close()

# Energy conservation

analytical_energy = 0.5 * mass * omega**2 * amplitude**2

energy_verlet = (0.5 * mass * omega**2 * position_verlet**2) + (0.5 * momentum_verlet**2 / mass)
energy_gear = (0.5 * mass * omega**2 * position_gear**2) + ( 0.5 * mass * velocity_gear**2) 

verlet_diff = (energy_verlet - analytical_energy) / analytical_energy
gear_diff = (energy_gear - analytical_energy) / analytical_energy
time = np.arange(0, total_time, timestep)

plt.figure(figsize=(12, 8))
plt.plot(time, np.full_like(time, analytical_energy), label="Analytical energy", linestyle='--', color='black')
plt.plot(time, energy_verlet, label="Verlet energy", color='blue')
plt.plot(time, energy_gear, label="Gear energy", color='green')
plt.xlabel('Time')
plt.ylabel('Energy')
plt.legend()
image_name = f"energy_comparison"
full_path = os.path.join(image_path, image_name)
plt.savefig(full_path)
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(time, verlet_diff)
plt.xlabel('Time')
plt.ylabel('Relative energy deviation')
plt.legend()
image_name = f"verlet_energy_dev"
full_path = os.path.join(image_path, image_name)
plt.savefig(full_path)
plt.show()
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(time, gear_diff)
plt.xlabel('Time')
plt.ylabel('Relative energy deviation')
plt.legend()
image_name = f"gear_energy_dev"
full_path = os.path.join(image_path, image_name)
plt.savefig(full_path)
plt.show()
plt.close()

# stability analysis

omega_list = [0.01, 1, 100] 
for index, omega in enumerate(omega_list):
    plt.figure(figsize=(10, 6))
    amplitude = np.sqrt(initial_condition_sq / omega**2)
    timestep = 0.001 / omega
    time = np.arange(0, total_time, timestep)
    an_position = analytical_position(amplitude, omega, time)
    an_velocity = analytical_velocity(amplitude, omega, time)
    position_verlet, momentum_verlet = md.velocity_verlet(initial_position, initial_momentum, harmonic_force, mass, omega, timestep, total_time)
    dev_pos = position_verlet - an_position
    dev_vel = (momentum_verlet / mass) - an_velocity
    plt.plot(time, dev_pos, label=f'Position Verlet deviation for $\\omega = {omega}$')
    plt.plot(time, dev_vel, label=f'Velocity Verlet deviation for $\\omega = {omega}$')
    plt.xlabel('Time')
    plt.ylabel('Deviation')
    plt.legend()
    image_name = f"stability_comparison_{index}"
    full_path = os.path.join(image_path, image_name)
    plt.savefig(full_path)
    plt.show()
    plt.close()