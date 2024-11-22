import numpy as np
import scipy.stats as stats
from collections.abc import Callable

def inversion(cdf_inverse: Callable[[float], float], number_sampled_points: int) -> np.ndarray:
    '''
    Samples a tuple of random variables using the inversion method from a probability density function that is 
    invertible and has as inverse cumulative density function cdf_inverse
    '''
    random_variables = np.zeros(number_sampled_points)
    for i in range(number_sampled_points):
        random_variables[i] = cdf_inverse(np.random.random())
    return random_variables    

def rejection_from_normal(target_distribution, c, mean, sigma, number_sampled_points):
    '''
    Samples a tuple of random variables using the rejection method from a probability density function \rho(x) stored 
    in an array called target_distribution, using as a candidate distribution the normal distribution N(x) with c satisfying 
    the inequality \rho(x) <= c N(x)
    '''
    counter = 0
    accepted_variables = np.zeros(number_sampled_points)
    while counter < number_sampled_points:
        candidate_random_variable = np.random.normal(mean, sigma)
        uniform_random_num = np.random.random()
        acceptance_threshold = target_distribution(candidate_random_variable)/ (c * stats.norm.pdf(candidate_random_variable, mean, sigma))
        if uniform_random_num <= acceptance_threshold:
            accepted_variables[counter] = candidate_random_variable
            counter+=1
    return accepted_variables     