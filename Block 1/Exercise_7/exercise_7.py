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
#INITIAL_MOLECULES = 250 # for a volume = 100
#INITIAL_MOLECULES = 2500 # for a volume = 1000
INITIAL_MOLECULES = 25000 # for a volume = 10000

X_INIT = INITIAL_MOLECULES
Y_INIT = INITIAL_MOLECULES
TIME_MAX = 100
INITIAL_CONFIGURATION = np.array([X_INIT,Y_INIT])

configurations, time_series = gillespie.algorithm(INITIAL_CONFIGURATION, TIME_MAX)

x_values, y_values = zip(*configurations)

x_values = np.array(x_values)
y_values = np.array(y_values)
time_series = np.array(time_series)


plt.rcParams.update({'font.size': 18})
plt.figure(figsize=(9, 6))
plt.scatter(x_values, y_values, color='blue', marker='.', s=1, alpha=0.4)  
plt.xlabel("X")
plt.ylabel("Y")
plt.title(f"Molecules graph for volume = {gillespie.VOLUME}")
image_name = f"gillespiexy_volume{gillespie.VOLUME}.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.close()


plt.figure(figsize=(15, 6))
plt.plot(time_series, x_values, color='blue', label = "X molecules")
plt.plot(time_series, y_values, color='red', label = "Y molecules")
plt.title(f"Molecules evolution graph for volume = {gillespie.VOLUME}")
plt.xlabel("Time")
plt.ylabel("Number of molecules")
plt.legend(fontsize = 10)
image_name = f"gillespiexy_time_volume{gillespie.VOLUME}.png"
entire_path = os.path.join(report_path, exercise_folder, image_name)
plt.savefig(entire_path)
plt.close()
