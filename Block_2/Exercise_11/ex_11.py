import numpy as np
import brownian as brown
import matplotlib.pyplot as plt

BOX_SIZE = 20
TIME_DIVISIONS = 10000
SIMULATION_TIME = 10

friction_coeff = 1
temperature = 1

force_list = np.zeros(TIME_DIVISIONS)

initial_position = [10,10,10]
harmonic_constant = 0

position_x = brown.first_order_integrator_component(initial_position[0], harmonic_constant, brown.force_harmonic_trap_component, temperature, friction_coeff, BOX_SIZE, TIME_DIVISIONS, SIMULATION_TIME)
position_y = brown.first_order_integrator_component(initial_position[1], harmonic_constant, brown.force_harmonic_trap_component, temperature, friction_coeff, BOX_SIZE, TIME_DIVISIONS, SIMULATION_TIME)
position_z = brown.first_order_integrator_component(initial_position[2], harmonic_constant, brown.force_harmonic_trap_component, temperature, friction_coeff, BOX_SIZE, TIME_DIVISIONS, SIMULATION_TIME)

# Plot trajectory
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(position_x[0], position_y[0], position_z[0], color='red', label='Initial Position')
ax.plot(position_x, position_y, position_z, label='Particle Trajectory', color='blue', alpha=0.7)
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
ax.set_zlabel("Z Position")
ax.set_title("3D Brownian Motion Trajectory")
ax.legend()
plt.show()