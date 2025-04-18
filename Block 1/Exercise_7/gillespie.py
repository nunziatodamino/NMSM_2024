import numpy as np
import numba

# parameters
VOLUME = 1e4
A = 2
B = 5

@numba.njit
def transition_rate_vector(configuration):
    '''
    Specific transition rates for the cyrcadian cycle
    '''
    transition_rate_1 = A * VOLUME
    def transition_rate_2(x):
        return x    
    def transition_rate_3(x, y):
        return (1 / (VOLUME**2)) * x * (x - 1) * y
    def transition_rate_4(x):
        return B * x
    rate_vector = np.array([transition_rate_1, transition_rate_2(configuration[0]), transition_rate_3(configuration[0], configuration[1]), transition_rate_4(configuration[0])])
    return rate_vector

@numba.njit
def escape_rate(configuration):
    '''
    Evaluation of the escape rate
    '''
    return np.sum(transition_rate_vector(configuration))

@numba.njit
def residence_time_cdf_inverse(configuration, cdf):
    '''
    CDF of the residence time PDF
    '''
    return 1 / escape_rate(configuration) * np.log(1 / cdf)

@numba.njit
def jump_function(old_configuration):
    '''
    Makes the transition between one state to another according to probability w_j/escape_rate
    '''
    rate_vector = transition_rate_vector(old_configuration)
    cumulative_sum = np.cumsum(rate_vector)
    rand = np.random.uniform(0, escape_rate(old_configuration))
    new_configuration = old_configuration.copy()

    if rand < cumulative_sum[0]: 
        new_configuration[0] += 1  # reaction 1
    elif rand < cumulative_sum[1]: 
        new_configuration[0] -= 1  # reaction 2
    elif rand < cumulative_sum[2]: 
        new_configuration[0] += 1  # reaction 3
        new_configuration[1] -= 1
    elif rand < cumulative_sum[3]: 
        new_configuration[0] -= 1  # reaction 4
        new_configuration[1] += 1

    return new_configuration

def algorithm(initial_configuration, time_max):
    '''
    Implementation of the Gillespie algorithm
    '''
    time_total = 0
    time_series = []
    configurations = [] 
    current_configuration = initial_configuration
    while time_total < time_max:
        residence_time = residence_time_cdf_inverse(current_configuration, np.random.random())
        time_total += residence_time
        time_series.append(time_total)
        current_configuration = jump_function(current_configuration)
        configurations.append(current_configuration)

    return configurations, time_series    