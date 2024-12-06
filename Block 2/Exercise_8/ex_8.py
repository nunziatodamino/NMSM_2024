import numpy as np
import off_lattice as md

BOX_SIZE = md.BOX_SIZE
SIGMA_CUT = md.SIGMA_CUT
EPSILON = md.EPSILON
SIGMA = md.SIGMA
PARTICLE_NUMBER = md.PARTICLE_NUMBER

# initialize positions and velocities
position_list = np.zeros((PARTICLE_NUMBER, 3))

for i in range(PARTICLE_NUMBER):
    position_list[i][0] = np.random.uniform(0, BOX_SIZE)
    position_list[i][1] = np.random.uniform(0, BOX_SIZE)
    position_list[i][2] = np.random.uniform(0, BOX_SIZE)

TEMPERATURE = 2
BETA = 1 / TEMPERATURE
ITERATIONS = 1000

for time in range(ITERATIONS):
    position_list = md.displacement_function(PARTICLE_NUMBER, BOX_SIZE, BETA, position_list)
