import numpy as np
import brownian as brown
import matplotlib.pyplot as plt

BOX_SIZE = 20
TIME_DIVISIONS = 1000
SIMULATION_TIME = 1
PARTICLE_NUMBER = 1000

force_list = np.zeros(TIME_DIVISIONS)
initial_position = [0,0,0]
time = np.linspace(0, SIMULATION_TIME, TIME_DIVISIONS)

harmonic_constant = 0

friction_coeff = 1
temperature_list = np.linspace(0.1, 2, 10)

for temp in temperature_list:
    print(f"iteration for temperature {temp}")
    msd = brown.brownian_motion(initial_position, harmonic_constant, brown.force_harmonic_trap_component, temp, friction_coeff, BOX_SIZE, TIME_DIVISIONS, SIMULATION_TIME, PARTICLE_NUMBER)
    plt.plot(time, msd, label=f"Temperature {temp}")

plt.xlabel('Time')
plt.ylabel('Mean square displacement')
plt.legend()
plt.show()

friction_list = np.linspace(10, 100, 10)
temperature = 1

for friction in friction_list:
    print(f"iteration for friction_coeff {friction}")
    msd = brown.brownian_motion(initial_position, harmonic_constant, brown.force_harmonic_trap_component, temperature, friction, BOX_SIZE, TIME_DIVISIONS, SIMULATION_TIME, PARTICLE_NUMBER)
    plt.plot(time, msd, label=f"Friction {friction}")

plt.xlabel('Time')
plt.ylabel('Mean square displacement')
plt.legend()
plt.show()

harmonic_constant_list = np.linspace(0.1, 10, 10)

for h_const in harmonic_constant_list:
    print(f"iteration for harmonic constant {h_const}")
    msd = brown.brownian_motion(initial_position, h_const, brown.force_harmonic_trap_component, temperature, friction_coeff, BOX_SIZE, TIME_DIVISIONS, SIMULATION_TIME, PARTICLE_NUMBER)
    plt.plot(time, msd, label=f"Harmonic constant {h_const}")

plt.xlabel('Time')
plt.ylabel('Mean square displacement')
plt.legend()
plt.show()

for _ in range(PARTICLE_NUMBER):
    _, position_x_unwrapped = brown.first_order_integrator_component(initial_position[0], harmonic_constant, brown.force_harmonic_trap_component, temperature, friction_coeff, BOX_SIZE, TIME_DIVISIONS, SIMULATION_TIME)
    plt.plot(time, position_x_unwrapped)

plt.xlabel('Time')
plt.ylabel('Position of the x component')
plt.legend()
plt.show()


# Plot trajectory
#fig = plt.figure(figsize=(10, 8))
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(position_x[0], position_y[0], position_z[0], color='red', label='Initial Position')
#ax.plot(position_x_un, position_y_un, position_z_un, label='Particle Trajectory', color='blue', alpha=0.7)
#ax.set_xlabel("X Position")
#ax.set_ylabel("Y Position")
#ax.set_zlabel("Z Position")
#ax.set_title("3D Brownian Motion Trajectory")
#ax.legend()
#plt.show()