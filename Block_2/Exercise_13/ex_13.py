import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import cell_list as md

SQUARE_SIZE = 100
PARTICLE_NUMBER_LIST = [2,10,50,50,500]
PARTICLE_RADIUS_1 = 1.25
PARTICLE_RADIUS_2 = 1.00
TIME_DIVISION = 1000
SIMULATION_TIME = 10000
FORCE_CONSTANT = 10

initial_position, radii_list = md.particles_initialization(PARTICLE_NUMBER_LIST[3], PARTICLE_RADIUS_1, PARTICLE_RADIUS_2, SQUARE_SIZE)
#md.plt_particles(initial_position, radii_list, PARTICLE_NUMBER_LIST[1], SQUARE_SIZE)

CELL_LENGTH = 4 * PARTICLE_RADIUS_1

neighbor_list = md.neighbor_list_square(SQUARE_SIZE, CELL_LENGTH)
mobility_list = 1 / radii_list

all_particle_positions = np.zeros((SIMULATION_TIME, PARTICLE_NUMBER_LIST[3], 2))
all_particle_positions[0] = initial_position

for t in range(1, SIMULATION_TIME):
    print(t)
    overlap = False
    cell_list, head_list = md.cell_linked_list(SQUARE_SIZE, CELL_LENGTH, all_particle_positions[t])
    force_x, force_y = md.force_evaluation(SQUARE_SIZE, CELL_LENGTH, PARTICLE_NUMBER_LIST[3], FORCE_CONSTANT, all_particle_positions[t], radii_list, neighbor_list, cell_list, head_list)
    attempt = 0
    for n in range(PARTICLE_NUMBER_LIST[3]):
        all_particle_positions[t][n][0] = md.first_order_integrator_component(all_particle_positions[t-1][n][0], force_x[n], mobility_list[n], SQUARE_SIZE, TIME_DIVISION, SIMULATION_TIME )
        all_particle_positions[t][n][1] = md.first_order_integrator_component(all_particle_positions[t-1][n][1], force_y[n], mobility_list[n], SQUARE_SIZE, TIME_DIVISION, SIMULATION_TIME )
        i, overlap = md.position_check(SQUARE_SIZE, CELL_LENGTH, all_particle_positions[t], radii_list, neighbor_list, cell_list, head_list)
        while(overlap):
            all_particle_positions[t][i][0] = md.first_order_integrator_component(all_particle_positions[t-1][i][0], force_x[n], mobility_list[n], SQUARE_SIZE, TIME_DIVISION, SIMULATION_TIME )
            all_particle_positions[t][i][1] = md.first_order_integrator_component(all_particle_positions[t-1][i][1], force_y[n], mobility_list[n], SQUARE_SIZE, TIME_DIVISION, SIMULATION_TIME )
            i, overlap = md.position_check(SQUARE_SIZE, CELL_LENGTH, all_particle_positions[t], radii_list, neighbor_list, cell_list, head_list)
        attempt +=1 
        if attempt % 100 == 0 :print(attempt)

md.plt_particles(all_particle_positions[1000], radii_list, PARTICLE_NUMBER_LIST[3], SQUARE_SIZE)

msd = np.zeros(SIMULATION_TIME)
for t in range(SIMULATION_TIME):
    displacements = all_particle_positions[t] - all_particle_positions[0]  
    squared_displacements = np.sum(displacements**2 , axis=1)  
    msd[t] = np.mean(squared_displacements)

# Plot MSD on a log-log scale
plt.figure(figsize=(8, 6))
plt.loglog(range(SIMULATION_TIME), msd, label="Mean Square Displacement")
plt.xlabel("Time Steps")
plt.ylabel("MSD")
plt.title("Mean Square Displacement of Particles (Log-Log Scale)")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

