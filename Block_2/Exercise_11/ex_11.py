import numpy as np
import os
import brownian as brown
import matplotlib.pyplot as plt

######################################################################
image_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block_2/Report/FIG/ex11"
######################################################################

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

plt.figure(figsize=(8, 6))
for temp in temperature_list:
    print(f"iteration for temperature {temp}")
    msd = brown.brownian_motion(initial_position, harmonic_constant, brown.force_harmonic_trap_component, temp, friction_coeff, BOX_SIZE, TIME_DIVISIONS, SIMULATION_TIME, PARTICLE_NUMBER)
    plt.plot(time, msd, label=f"$T^* = {temp:.1f}$")
plt.xlabel('Time')
plt.ylabel('Mean square displacement')
plt.legend()
image_name = f"msd_temp.png"
full_path = os.path.join(image_path, image_name)
plt.savefig(full_path)
plt.show()
plt.close()

friction_list = np.linspace(10, 100, 10)
temperature = 1

plt.figure(figsize=(8, 6))
for friction in friction_list:
    print(f"iteration for friction_coeff {friction}")
    msd = brown.brownian_motion(initial_position, harmonic_constant, brown.force_harmonic_trap_component, temperature, friction, BOX_SIZE, TIME_DIVISIONS, SIMULATION_TIME, PARTICLE_NUMBER)
    plt.plot(time, msd, label=f"$ \gamma \\tau = {friction:.1f}$")
plt.xlabel('Time')
plt.ylabel('Mean square displacement')
plt.legend()
image_name = f"msd_friction.png"
full_path = os.path.join(image_path, image_name)
plt.savefig(full_path)
plt.show()
plt.close()

harmonic_constant_list = np.linspace(0.1, 10, 10)

plt.figure(figsize=(8, 6))
for h_const in harmonic_constant_list:
    print(f"iteration for harmonic constant {h_const}")
    msd = brown.brownian_motion(initial_position, h_const, brown.force_harmonic_trap_component, temperature, friction_coeff, BOX_SIZE, TIME_DIVISIONS, SIMULATION_TIME, PARTICLE_NUMBER)
    plt.plot(time, msd, label=f"$ K\sigma^2/ \\varepsilon =  {h_const:.1f}$")
plt.xlabel('Time')
plt.ylabel('Mean square displacement')
plt.legend()
image_name = f"mds_hconst.png"
full_path = os.path.join(image_path, image_name)
plt.savefig(full_path)
plt.show()
plt.close()

plt.figure(figsize=(8, 6))
for _ in range(PARTICLE_NUMBER):
    _, position_x_unwrapped = brown.first_order_integrator_component(initial_position[0], harmonic_constant, brown.force_harmonic_trap_component, temperature, friction_coeff, BOX_SIZE, TIME_DIVISIONS, SIMULATION_TIME)
    plt.plot(time, position_x_unwrapped)
plt.xlabel('Time')
plt.ylabel('Position of the x component')
image_name = f"x_component.png"
full_path = os.path.join(image_path, image_name)
plt.savefig(full_path)
plt.show()
plt.close()

harmonic_constant = 10

plt.figure(figsize=(8, 6))
for _ in range(PARTICLE_NUMBER):
    _, position_x_unwrapped = brown.first_order_integrator_component(initial_position[0], harmonic_constant, brown.force_harmonic_trap_component, temperature, friction_coeff, BOX_SIZE, TIME_DIVISIONS, SIMULATION_TIME)
    plt.plot(time, position_x_unwrapped)
plt.xlabel('Time')
plt.ylabel('Position of the x component')
image_name = f"x_component_harm.png"
full_path = os.path.join(image_path, image_name)
plt.savefig(full_path)
plt.show()
plt.close()

# Plot trajectory
#fig = plt.figure(figsize=(10, 8))
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(position_x[0], position_y[0], position_z[0], color='red', label='Initial Position')
#ax.plot(position_x_un, position_y_un, position_z_un, label='Particle Trajectory', color='blue', alpha=0.7)
#ax.set_xlabel("X")
#ax.set_ylabel("Y")
#ax.set_zlabel("Z")
#ax.legend()
#plt.show()