import numpy as np
import matplotlib.pyplot as plt
import distribution_sampling as sampling

def pdf_normalized_1 (x):
    return 2 * x *np.exp(-x**2)

def pdf_normalized_2 (x):
    return 5 / 243 * x ** 4

def cdf_inverse_1 (y):
    return np.sqrt(np.log(1/(1-y)))

def cdf_inverse_2 (y):
    return (243 * y)**(1/5)

NUMBER_SAMPLED_POINTS = 100000

random_variables_1 = sampling.inversion_sampling(cdf_inverse_1, NUMBER_SAMPLED_POINTS)
random_variables_2 = sampling.inversion_sampling(cdf_inverse_2, NUMBER_SAMPLED_POINTS)

sample_space_1 = np.arange(0, 4, 0.01)
pdf1 = pdf_normalized_1(sample_space_1)

plt.plot(sample_space_1, pdf1, marker='.', linestyle='-', color='b')
plt.hist(random_variables_1, bins=40, density=True, alpha=0.4, color='r')

plt.title('First pdf')
plt.show()

sample_space_2 = np.arange(0, 3, 0.01)
pdf2 = pdf_normalized_2(sample_space_2)

plt.plot(sample_space_2, pdf2, marker='.', linestyle='-', color='b')
plt.hist(random_variables_2, bins=30, density=True, alpha=0.4, color='r')

plt.title('Second pdf')
plt.show()
