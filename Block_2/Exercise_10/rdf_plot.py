import numpy as np
import matplotlib.pyplot as plt

# Parameters
filename = '/home/omega/Documents/NMSM/Block_2/Exercise_10/rdf_1.12.dat'  # Replace with your actual RDF file path
n = 4  # The number of columns in the .rdf file
nn = 150000  # The total number of timesteps in your .rdf file

# Read the RDF file
with open(filename, 'r') as fid:
    # Skip the first three lines
    for _ in range(3):
        next(fid)
    
    # Read the rest of the data
    data = np.loadtxt(fid)

# Extract the number of bins (m)
m = int(data[0, 1])  # Assuming the second column in the first row gives the number of bins

# Initialize arrays
gr = np.zeros(m)
pos = np.zeros(m)

# Process the RDF data
for i in range(nn):
    # Calculate the starting index for the current timestep
    start_idx = i * (2 + m * n) + n  # MATLAB index starts at 1, so adjust accordingly
    
    # Extract the position and RDF values
    pos = data[start_idx + 2 + n * np.arange(m), 0]
    grl = data[start_idx + 3 + n * np.arange(m), 1]
    
    # Sum the RDF values over timesteps
    gr += grl

# Normalize the RDF
gr /= nn

# Plot the RDF
plt.figure(figsize=(8, 6))
plt.plot(pos, gr)
plt.xlabel('Radial Distance (r)')
plt.ylabel('g(r)')
plt.title('Radial Distribution Function')
plt.grid(True)
plt.show()
