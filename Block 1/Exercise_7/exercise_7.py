# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import gillespie as gillespie

# paths for saving images in report
########################################
report_path = "/home/nunziato-damino/Documents/Github/NMSM_2024/Block 1/Report"
exercise_folder = "FIG/exercise_7_images"
########################################

# initial state
X_INIT = 2000
Y_INIT = 2000
TIME_MAX = 100
INITIAL_CONFIGURATION = np.array([X_INIT,Y_INIT])

configurations, time_series = gillespie.algorithm(INITIAL_CONFIGURATION, TIME_MAX)

x_values, y_values = zip(*configurations)

x_values = np.array(x_values)
y_values = np.array(y_values)
time_series = np.array(time_series)


# %%

plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(10, 6))
plt.scatter(x_values, y_values, color='blue', marker='.', s=1, alpha=0.4)  
plt.xlabel("X")
plt.ylabel("Y")
#plt.title("Gillespie Simulation Results")
image_name = "gillespiexy.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.show()


plt.plot(time_series, x_values, color='blue', marker='.', label = "X molecules")
plt.plot(time_series, y_values, color='red', marker='.', label = "Y molecules")

plt.xlabel("Time")
plt.ylabel("Number of molecules")
plt.legend()
#plt.title("Gillespie Simulation Results")
#image_name = "gillespiexy.png"
#entire_path = os.path.join(report_path, exercise_folder, image_name)
#plt.savefig(entire_path)
#plt.close()
plt.show()

# %%