import matplotlib.pyplot as plt
import numpy as np

iterations = [50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000, 130000, 140000, 150000, 160000, 170000, 180000, 190000, 200000]
totenergy_r_low = [0.31138112, 0.27140862, 0.33140508, 0.3314411, 0.35145159, 0.34148605, 0.26650524, 0.286487, 0.36144296, 0.36647086, 0.3214366, 0.32141597, 0.33144332, 0.30144998, 0.30141907, 0.29639279]
totenergy_r_high = [0.61393148, 0.61442332, 0.61413377, 0.61397173, 0.61395918, 0.61407106, 0.61399433, 0.61351255, 0.61452365, 0.61363972, 0.61302539, 0.61344442, 0.61414501, 0.61348899, 0.61364329, 0.61316784]

plt.figure(figsize=(10, 6))
plt.plot(iterations, totenergy_r_low, marker='o', linestyle='-', color='b', label=r'Total energy for cutoff radius = 2^(1/6) \sigma')
plt.plot(iterations, totenergy_r_high, marker='o', linestyle='-', color='r', label=r'Total energy for cutoff radius = 4 \sigma')

plt.xlabel('Iteration', fontsize=14)
plt.ylabel('Energy time series', fontsize=14)
plt.ylim(0,0.7)
plt.legend(fontsize=12)
plt.tight_layout()
plt.show()
