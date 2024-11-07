import numpy as np
import matplotlib.pyplot as plt
import gillespie as gillespie

# initial state
X_INIT = 100
Y_INIT = 100
INITIAL_CONFIGURATION = np.array([X_INIT,Y_INIT])

conf = gillespie.algorithm(INITIAL_CONFIGURATION, 100)

x_values, y_values = zip(*conf)

plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='blue', marker='.', s=1, alpha=0.4)  
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Gillespie Simulation Results")
plt.show()