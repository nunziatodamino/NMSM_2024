import matplotlib.pyplot as plt
import numpy as np
import os
import re

directory = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block_2/Exercise_10/NVE_simulation"
file_pattern = re.compile(r'rdf_(\d+\.\d+)\.dat')
file_names = [f for f in os.listdir(directory) if file_pattern.match(f)]

plt.figure(figsize=(10, 6))

for file_name in sorted(file_names):
    file_path = os.path.join(directory, file_name)
    r_cut = file_pattern.match(file_name).group(1)
    
    with open(file_path, 'r') as file:
        for _ in range(4):
            next(file)
        
        accumulated_g_r = None
        block_count = 0
        r_values = None
        
        for i in range(1000):
            lines = [file.readline() for _ in range(200)]
            if not lines or len(lines) < 200:
                break

            data = np.array([list(map(float, line.split())) for line in lines if line.strip()])
            
            if r_values is None:
                r_values = data[:, 1]
            
            g_r_values = data[:, 2]
            
            if accumulated_g_r is None:
                accumulated_g_r = g_r_values
            else:
                accumulated_g_r += g_r_values
            
            block_count += 1
            file.readline()
        
        averaged_g_r = accumulated_g_r / block_count
        plt.plot(r_values, averaged_g_r, label=f'r_cut = {r_cut}')

plt.xlabel("r")
plt.ylabel("g(r)")
plt.legend()
plt.grid(True)
plt.show()
