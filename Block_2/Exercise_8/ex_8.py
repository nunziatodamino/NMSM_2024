import numpy as np
import matplotlib.pyplot as plt
import off_lattice as md

######################################################################
file_path = "/home/omega/Documents/NMSM/Block_2/Exercise_8/LJ_T09.dat"
#file_path = "/home/omega/Documents/NMSM/Block_2/Exercise_8/LJ_T2.dat"
######################################################################

EPSILON = md.EPSILON
SIGMA = md.SIGMA
TEMPERATURE = 0.9
#TEMPERATURE = 2
BETA = 1 / TEMPERATURE
ITERATIONS = 10000

particle_number = 100
data_points = 10
number_density_list = np.linspace(0.1, 0.9, data_points)
box_size_list = (particle_number / number_density_list) ** (1/3)
pressure = np.zeros(data_points)

# initialize positions and do evolution
for n, box_size in enumerate(box_size_list):
    sigma_cut = box_size / 2
    print(f"Iteration for number density : {number_density_list[n]}")
    position_list = np.zeros((particle_number, 3), dtype=np.float64)
    for i in range(particle_number):
        position_list[i][0] = np.random.uniform(0, box_size)
        position_list[i][1] = np.random.uniform(0, box_size)
        position_list[i][2] = np.random.uniform(0, box_size)
    counter = 0
    for time in range(ITERATIONS):
        partial_counter, position_list = md.displacement_function(particle_number, number_density_list[n], box_size, sigma_cut, BETA, position_list)
        counter += partial_counter
    print(f"acceptance rate {(counter / (ITERATIONS * particle_number ) )* 100 } %")    
    pressure[n] = md.pressure(number_density_list[n], box_size, TEMPERATURE, EPSILON, SIGMA, sigma_cut, position_list)    

md.plot_particles(position_list, box_size_list[-1])
data = np.loadtxt(file_path, delimiter=' ')
number_density_to_check = data[:, 0]
pressure_to_check = data[:, 1]
plt.figure(figsize=(8, 6))
plt.plot(number_density_to_check, pressure_to_check, marker='o', linestyle='-', label="Data")
plt.plot(number_density_list, pressure, marker='o', linestyle='-', label="Simulation" )
plt.xlabel('Number Density')
plt.ylabel('Pressure')
plt.title('Number Density vs. Pressure')
plt.grid(True)
plt.legend()
plt.show()