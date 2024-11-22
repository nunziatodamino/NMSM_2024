import numpy as np
import scipy.stats as stats

def crude_monte_carlo_uniform (function, left_bound, right_bound, iterations):
    '''
    Evaluates the integral of a function contained in the array function from the left_bound to the right_bound 
    sampling from the uniform distribution between these bounds
    '''
    values = np.random.uniform(left_bound, right_bound, size = iterations)
    func_values = function(values)
    estimate = (right_bound - left_bound ) * np.sum(func_values) / iterations
    variance = (right_bound - left_bound )**2 * np.sum(func_values**2) / iterations - estimate**2
    return estimate, np.sqrt( variance / iterations ) 

def crude_monte_carlo_normal (function, mean, sigma, iterations):
    '''
    Evaluates the integral of a function contained in the array function from the left_bound to the right_bound 
    sampling from the normal distribution between these bounds
    '''
    values = np.random.normal(mean, sigma, size = iterations)
    func_values = function(values)
    estimate = np.sum(func_values) / iterations
    variance = np.sum(func_values**2) / iterations - estimate**2
    return estimate, np.sqrt(variance / iterations)

def truncated_normal (x ,mean, sigma, left_t, right_t):
    '''
    Defines a truncated normal of given mean, sigma, and truncation at left_t and right_t
    '''
    return stats.norm.pdf(x, mean, sigma) / (stats.norm.cdf(right_t) - stats.norm.cdf(left_t))

def importance_sampling_mc_vectorized(integrand_function, sampling_function, number_sampled_points):
    '''
    ATTENTION : specialized function
    This code evaluates the integral in the proposed exercise through importance sampling with the function sampling_function
    contained in an array. The code contains the sampling from this distribution through a rejection sampling using a truncated 
    normal an parameter c. Moreover the code is vectorized for speed. 
    '''
    left_bound  = 0
    right_bound = np.pi / 2
    mean = np.pi/2
    sigma = 1
    c = 1.4
    a1, b1 = (left_bound - mean) / sigma, (right_bound - mean) / sigma
    batch_size = 10 * number_sampled_points 
    random_variables = stats.truncnorm.rvs(a1, b1, loc=mean, scale=sigma, size=batch_size)
    acceptance_thresholds = sampling_function(random_variables) / (c * truncated_normal(random_variables, mean, sigma, left_bound, right_bound))       
    uniform_random_nums = np.random.random(batch_size)
    accepted_indices = uniform_random_nums <= acceptance_thresholds # boolean indexing of uniform_random_numbers
    accepted_samples = random_variables[accepted_indices]
    accepted_samples = accepted_samples[accepted_samples != 0] # remove extra zeros
    if len(accepted_samples) > number_sampled_points:
        accepted_samples = accepted_samples[:number_sampled_points] 
    func_values = integrand_function(accepted_samples) / sampling_function(accepted_samples)    
    estimate = np.sum(func_values) / number_sampled_points                
    variance = np.sum(func_values**2) / number_sampled_points - estimate**2
    return estimate , np.sqrt(variance / number_sampled_points)
