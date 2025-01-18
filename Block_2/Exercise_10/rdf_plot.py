import matplotlib.pyplot as plt
import numpy as np

file_name = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block_2/Exercise_10/rdf_4.12.dat"

with open(file_name, 'r') as file:
    for _ in range(4):
        next(file)
    
    accumulated_data = None

    for i in range(100):
        lines = [file.readline() for _ in range(200)]
        if not lines or len(lines) < 200:  # If we run out of lines
            break

        data = np.array([list(map(float, line.split())) for line in lines])
        if accumulated_data is None:
            accumulated_data = data
        else:
            accumulated_data += data
        file.readline()
averaged_data = accumulated_data / 100  

plt.plot(averaged_data[:, 1], averaged_data[:, 2])

plt.xlabel("r")
plt.ylabel("g(r)")
plt.show()
