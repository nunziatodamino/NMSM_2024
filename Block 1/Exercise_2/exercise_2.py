import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv
import distribution_sampling as sampling

A = np.sqrt(2)/(np.e * kv(1/4,1))

def non_invertible_pdf (x):
    return A * np.exp ( -8*(x**2/2+x**4/4) )

mean = 0
sigma = np.sqrt(1/4)
c = 1.6

NUMBER_SAMPLED_VARIABLES = 10000

random_variables = sampling.rejection_from_normal(non_invertible_pdf, c, mean, sigma, NUMBER_SAMPLED_VARIABLES)

x = np.arange(-3 ,+3, 0.01)
y = non_invertible_pdf(x)

plt.rcParams.update({'font.size': 18})

plt.plot (x,y, label = "Target distribution")
plt.hist(random_variables, bins=30, density=True, alpha=0.4, color='r', label= "Sampled variables")
plt.xlabel("Random variable")
plt.ylabel("Probability density")
plt.legend()
plt.show()