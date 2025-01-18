import numpy as np
import os
import matplotlib.pyplot as plt
import cell_list as md

######################################################################
image_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block_2/Report/FIG/ex13"
######################################################################

SQUARE_SIZE = 100
PARTICLE_NUMBER_LIST = [2,10,50,200,500]
PARTICLE_RADIUS_1 = 1.25
PARTICLE_RADIUS_2 = 1.00
TIME_DIVISION = 10000
SIMULATION_TIME = 10000
FORCE_CONSTANT = 10

for particle_number in PARTICLE_NUMBER_LIST:
    number_density = particle_number / (SQUARE_SIZE**2)
    print(f"Iteration for rho = {number_density}")

    initial_position, radii_list = md.particles_initialization(particle_number, PARTICLE_RADIUS_1, PARTICLE_RADIUS_2, SQUARE_SIZE)
    #md.plt_particles(initial_position, radii_list, PARTICLE_NUMBER_LIST[1], SQUARE_SIZE)

    CELL_LENGTH = 8 * PARTICLE_RADIUS_1

    neighbor_list = md.neighbor_list_square(SQUARE_SIZE, CELL_LENGTH)
    mobility_list = 1 / radii_list

    all_particle_positions = np.zeros((SIMULATION_TIME, particle_number, 2))
    all_particle_positions[0] = initial_position

    for t in range(1, SIMULATION_TIME):
        print(t)
        overlap = False
        cell_list, head_list = md.cell_linked_list(SQUARE_SIZE, CELL_LENGTH, all_particle_positions[t])
        force_x, force_y = md.force_evaluation(SQUARE_SIZE, CELL_LENGTH, particle_number, FORCE_CONSTANT, all_particle_positions[t], radii_list, neighbor_list, cell_list, head_list)
        attempt = 0
        for n in range(particle_number):
                all_particle_positions[t][n][0] = md.first_order_integrator_component(all_particle_positions[t-1][n][0], force_x[n], mobility_list[n], SQUARE_SIZE, TIME_DIVISION, SIMULATION_TIME )
                all_particle_positions[t][n][1] = md.first_order_integrator_component(all_particle_positions[t-1][n][1], force_y[n], mobility_list[n], SQUARE_SIZE, TIME_DIVISION, SIMULATION_TIME )
        #cell_list, head_list = md.cell_linked_list(SQUARE_SIZE, CELL_LENGTH, all_particle_positions[t])
        #index_list, overlap = md.position_check(SQUARE_SIZE, CELL_LENGTH, all_particle_positions[t], radii_list, neighbor_list, cell_list, head_list)
        index, overlap = md.check_overlap(all_particle_positions[t], radii_list)
        while(overlap):
            all_particle_positions[t][index][0] = md.first_order_integrator_component(all_particle_positions[t-1][index][0], force_x[n], mobility_list[n], SQUARE_SIZE, TIME_DIVISION, SIMULATION_TIME )
            all_particle_positions[t][index][1] = md.first_order_integrator_component(all_particle_positions[t-1][index][1], force_y[n], mobility_list[n], SQUARE_SIZE, TIME_DIVISION, SIMULATION_TIME )
            #cell_list, head_list = md.cell_linked_list(SQUARE_SIZE, CELL_LENGTH, all_particle_positions[t])
            #index_list, overlap = md.position_check(SQUARE_SIZE, CELL_LENGTH, all_particle_positions[t], radii_list, neighbor_list, cell_list, head_list)
            index, overlap = md.check_overlap(all_particle_positions[t], radii_list)
            attempt +=1 
            if attempt % 10000 == 0 :print(f"Attempt : {attempt}")

    #md.plt_particles(all_particle_positions[5000], radii_list, particle_number, SQUARE_SIZE)

    msd_x = np.zeros(SIMULATION_TIME)
    msd_x_particle1 = np.zeros(SIMULATION_TIME)
    msd_x_particle2 = np.zeros(SIMULATION_TIME)
    for t in range(SIMULATION_TIME):
        displacements_x = all_particle_positions[t][:, 0] - all_particle_positions[0][:, 0]
        displacements_x_particle1 = all_particle_positions[t][:particle_number // 2, 0] - all_particle_positions[0][:particle_number // 2, 0] 
        displacements_x_particle2 = all_particle_positions[t][particle_number // 2:, 0] - all_particle_positions[0][particle_number // 2:, 0] 
        squared_displacements_x = displacements_x**2 
        squared_displacements_x_particle1 = displacements_x_particle1**2 
        squared_displacements_x_particle2 = displacements_x_particle2**2 
        msd_x[t] = np.mean(squared_displacements_x)
        msd_x_particle1[t] = np.mean(squared_displacements_x_particle1)
        msd_x_particle2[t] = np.mean(squared_displacements_x_particle2)

    plt.figure(figsize=(8, 6))
    plt.loglog(range(SIMULATION_TIME), msd_x, label="Mean square displacement")
    plt.xlabel("Time steps")
    plt.ylabel("MSD")
    plt.title("Mean square Displacement of particles along x (log-log Scale)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    image_name = f"MSD_particle_number{particle_number}"
    full_path = os.path.join(image_path, image_name)
    plt.savefig(full_path)
    plt.show()
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.loglog(range(SIMULATION_TIME), msd_x_particle1, label="Mean square displacement for particle 1")
    plt.loglog(range(SIMULATION_TIME), msd_x_particle2, label="Mean square displacement for particle 2")
    plt.xlabel("Time steps")
    plt.ylabel("MSD")
    plt.title("Mean square displacement of particles along x (log-log Scale)")
    plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    image_name = f"MSD_particle_number{particle_number}_particle1"
    full_path = os.path.join(image_path, image_name)
    plt.savefig(full_path)
    plt.show()
    plt.close()